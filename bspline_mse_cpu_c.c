/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "bspline.h"
#include "bspline_mse_cpu_c.h"
#include "timer.h"

/* Mean-squared error version of implementation "C" */
/* Implementation "C" is slower than "B", but yields a smoother cost function 
   for use by L-BFGS-B.  It uses linear interpolation of moving image, 
   and nearest neighbor interpolation of gradient */
void
bspline_score_c_mse (
    BSPLINE_Parms *parms, 
    Bspline_state *bst,
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int rijk[3];             /* Indices within fixed image region (vox) */
    int fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];           /* Indices within moving image (vox) */
    float fxyz[3];           /* Position within fixed image (mm) */
    float mxyz[3];           /* Position within moving image (mm) */
    int mijk_f[3], mvf;      /* Floor */
    int mijk_r[3], mvr;      /* Round */
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    Timer timer;
    double interval;
    float m_val;

    /* GCS: Oct 5, 2009.  We have determined that sequential accumulation
       of the score requires double precision.  However, reduction 
       accumulation does not. */
    double score_acc = 0.;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "wb");
    }

    plm_timer_start (&timer);

    ssd->score = 0.0f;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
    for (rijk[2] = 0, fijk[2] = bxf->roi_offset[2]; rijk[2] < bxf->roi_dim[2]; rijk[2]++, fijk[2]++) {
	p[2] = rijk[2] / bxf->vox_per_rgn[2];
	q[2] = rijk[2] % bxf->vox_per_rgn[2];
	fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];
	for (rijk[1] = 0, fijk[1] = bxf->roi_offset[1]; rijk[1] < bxf->roi_dim[1]; rijk[1]++, fijk[1]++) {
	    p[1] = rijk[1] / bxf->vox_per_rgn[1];
	    q[1] = rijk[1] % bxf->vox_per_rgn[1];
	    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
	    for (rijk[0] = 0, fijk[0] = bxf->roi_offset[0]; rijk[0] < bxf->roi_dim[0]; rijk[0]++, fijk[0]++) {
		int rc;
		p[0] = rijk[0] / bxf->vox_per_rgn[0];
		q[0] = rijk[0] % bxf->vox_per_rgn[0];
		fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];

		/* Get B-spline deformation vector */
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

		/* Compute moving image coordinate of fixed image voxel */
		rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
						  dxyz, moving);

		/* If voxel is not inside moving image */
		if (!rc) {
		    continue;
		}

		/* Compute interpolation fractions */
		CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
					     li_1, li_2, moving);

		/* Find linear index of "corner voxel" in moving image */
		mvf = INDEX_OF (mijk_f, moving->dim);

		/* Compute moving image intensity using linear interpolation */
		/* Macro is slightly faster than function */
		BSPLINE_LI_VALUE (m_val, 
				  li_1[0], li_2[0],
				  li_1[1], li_2[1],
				  li_1[2], li_2[2],
				  mvf, m_img, moving);

		/* Compute linear index of fixed image voxel */
		fv = INDEX_OF (fijk, fixed->dim);

		/* Compute intensity difference */
		diff = m_val - f_img[fv];

		/* Compute spatial gradient using nearest neighbors */
		mvr = INDEX_OF (mijk_r, moving->dim);
		dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
		bspline_update_grad_b (bst, bxf, pidx, qidx, dc_dv);
		
		if (parms->debug) {
		    fprintf (fp, "%d %d %d %g %g %g\n", 
			     rijk[0], rijk[1], rijk[2], 
			     dc_dv[0], dc_dv[1], dc_dv[2]);
		}
		score_acc += diff * diff;
		num_vox ++;
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->score = score_acc / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}
