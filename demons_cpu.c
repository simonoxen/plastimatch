/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "readmha.h"
#include "volume.h"
#include "demons_opts.h"
#include "demons_misc.h"

int
round_int (float f)
{
    return (f > 0) ? (int)(f+0.5) : (int)(ceil(f-0.5));
}

/* Vector fields are all in mm units */
Volume*
demons_c (Volume* fixed, Volume* moving, Volume* moving_grad, Volume* vf_init, DEMONS_Parms* parms)
{
    int i, j, k, v;
    int	it;			    /* Iterations */
    int fi, fj, fk, fv;		    /* Indices into fixed image */
    float mi, mj, mk;		    /* Indices into moving image */
    float f2mo[3];		    /* Offset difference (in cm) from fixed to moving */
    float f2ms[3];		    /* Slope to convert fixed to moving */
    float invmps[3];		    /* 1/pixel spacing of moving image */
    float *kerx, *kery, *kerz;
    float *dxyz;
    int mv, mx, my, mz;
    int fw[3];
    clock_t start_run, end_run;
    double diff_run;
    Volume *vf_est, *vf_smooth;
    Volume *m_grad_mag;
    float *f_img = (float*) fixed->img;
    float *m_img = (float*) moving->img;
    float *m_grad_img = (float*) moving_grad->img;
    float *m_grad_mag_img;
    float *vox_grad, vox_grad_mag;
    float diff, mult, denom;
    float *vf_est_img, *vf_smooth_img;
    int inliers;
    float ssd;

    /* Allocate memory for vector fields */
    if (vf_init) {
	/* If caller has an initial estimate, we copy it */
	vf_smooth = volume_clone (vf_init);
	vf_convert_to_interleaved (vf_smooth);
    } else {
	/* Otherwise initialize to zero */
	vf_smooth = volume_create (fixed->dim, fixed->offset, fixed->pix_spacing, PT_VF_FLOAT_INTERLEAVED, 0);
    }
    vf_est = volume_create (fixed->dim, fixed->offset, fixed->pix_spacing, PT_VF_FLOAT_INTERLEAVED, 0);
    vf_est_img = (float*) vf_est->img;
    vf_smooth_img = (float*) vf_smooth->img;
    m_grad_mag = volume_create (moving->dim, moving->offset, moving->pix_spacing, PT_FLOAT, 0);
    m_grad_mag_img = (float*) m_grad_mag->img;

    /* Create gradient magnitude image */
    for (v = 0, k = 0; k < moving->dim[2]; k++) {
	for (j = 0; j < moving->dim[1]; j++) {
	    for (i = 0; i < moving->dim[0]; i++, v++) {
		vox_grad = &m_grad_img[3*v];
		m_grad_mag_img[v] = vox_grad[0] * vox_grad[0] + vox_grad[1] * vox_grad[1] + vox_grad[2] * vox_grad[2];
	    }
	}
    }

    /* Validate filter widths */
    validate_filter_widths (fw, parms->filter_width);

    /* Create the seperable smoothing kernels for the x, y, and z directions */
    kerx = create_ker (parms->filter_std / fixed->pix_spacing[0], fw[0]/2);
    kery = create_ker (parms->filter_std / fixed->pix_spacing[1], fw[1]/2);
    kerz = create_ker (parms->filter_std / fixed->pix_spacing[2], fw[2]/2);
    kernel_stats (kerx, kery, kerz, fw);

    /* Compute some variables for converting pixel sizes / offsets */
    for (i = 0; i < 3; i++) {
	invmps[i] = 1 / moving->pix_spacing[i];
	f2mo[i] = (fixed->offset[i] - moving->offset[i]) / moving->pix_spacing[i];
	f2ms[i] = fixed->pix_spacing[i] / moving->pix_spacing[i];
    }

    start_run = clock();

    /* Main loop through iterations */
    for (it = 0; it < parms->max_its; it++) {

	/* Estimate displacement, store into vf_est */
	inliers = 0; ssd = 0.0;
	memcpy (vf_est_img, vf_smooth_img, vf_est->npix * vf_est->pix_size);
	for (fv = 0, fk = 0, mk = f2mo[2]; fk < fixed->dim[2]; fk++, mk+=f2ms[2]) {
	    for (fj = 0, mj = f2mo[1]; fj < fixed->dim[1]; fj++, mj+=f2ms[1]) {
		for (fi = 0, mi = f2mo[0]; fi < fixed->dim[0]; fi++, mi+=f2ms[0], fv++) {

		    /* Find correspondence with nearest neighbor interpolation & boundary checking */
		    dxyz = &vf_smooth_img [3*fv];			/* mm */
		    mz = round_int(mk + invmps[2] * dxyz[2]);		/* pixels (moving) */
		    if (mz < 0 || mz >= moving->dim[2]) continue;
		    my = round_int(mj + invmps[1] * dxyz[1]);		/* pixels (moving) */
		    if (my < 0 || my >= moving->dim[1]) continue;
		    mx = round_int(mi + invmps[0] * dxyz[0]);		/* pixels (moving) */
		    if (mx < 0 || mx >= moving->dim[0]) continue;
		    mv = (mz * moving->dim[1] + my) * moving->dim[0] + mx;

		    /* Find image difference at this correspondence */
		    diff = f_img[fv] - m_img[mv];			/* intensity */

		    /* Find spatial gradient at this correspondece */
		    vox_grad = &m_grad_img[3*mv];			/* intensity per mm */
		    vox_grad_mag = m_grad_mag_img[mv];			/* intensity^2 per mm^2 */

		    /* Compute denominator */
		    denom = vox_grad_mag + parms->homog * diff * diff;	/* intensity^2 per mm^2 */

		    /* Compute SSD for statistics */
		    inliers++;
		    ssd += diff * diff;

		    /* Threshold the denominator to stabilize estimation */
		    if (denom < parms->denominator_eps) continue;

		    /* Compute new estimate of displacement */
		    mult = parms->accel * diff / denom;			/* per intensity^2 */
		    vf_est_img[3*fv + 0] += mult * vox_grad[0];		/* mm */
		    vf_est_img[3*fv + 1] += mult * vox_grad[1];
		    vf_est_img[3*fv + 2] += mult * vox_grad[2];
		}
	    }
	}
	printf ("----- SSD = %.01f (%d/%d)\n", ssd/inliers, inliers, fixed->npix);
	//vf_print_stats (vf_est);

	/* Smooth the estimate into vf_smooth.  The volumes are ping-ponged. */
	vf_convolve_x (vf_smooth, vf_est, kerx, fw[0]);
	vf_convolve_y (vf_est, vf_smooth, kery, fw[1]);
	vf_convolve_z (vf_smooth, vf_est, kerz, fw[2]);
	//vf_print_stats (vf_smooth);
    }

    free (kerx);
    free (kery);
    free (kerz);
    volume_free (vf_est);
    volume_free (m_grad_mag);

    end_run = clock();
    diff_run = (double)(end_run - start_run)/CLOCKS_PER_SEC;
    printf ("Time to estimate vector field for %d iterations = %f (%f sec / it)\n", 
	    parms->max_its, diff_run, diff_run / parms->max_its);

    return vf_smooth;
}

