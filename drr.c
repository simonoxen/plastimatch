/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#include "math_util.h"
#include "drr.h"
#include "drr_cuda.h"
#include "drr_opts.h"
#include "drr_trilin.h"
#include "proj_image.h"
#include "proj_matrix.h"
#include "ray_trace_exact.h"
#include "readmha.h"
#include "volume_limit.h"
#include "timer.h"

typedef struct callback_data Callback_data;
struct callback_data {
    double accum;               /* Accumulated intensity */
    int num_pix;                /* Number of pixels traversed */
};


//#define ULTRA_VERBOSE 1
//#define VERBOSE 1

/* According to NIST, the mass attenuation coefficient of H2O at 50 keV
   is 0.22 cm^2 per gram.  Thus, we scale by 0.022 per mm
   http://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html  */
static float
attenuation_lookup_hu (float pix_density)
{
    const double min_hu = -800.0;
    const double mu_h2o = 0.022;
    if (pix_density <= min_hu) {
	return 0.0;
    } else {
	return (pix_density/1000.0) * mu_h2o + mu_h2o;
    }
}

static float
attenuation_lookup (float pix_density)
{
    return attenuation_lookup_hu (pix_density);
}

void
drr_preprocess_attenuation (Volume* vol)
{
    int i;
    float* new_img;
    float* old_img;

    old_img = (float*) vol->img;
    new_img = (float*) malloc (vol->npix*sizeof(float));
    
    for (i = 0; i < vol->npix; i++) {
	new_img[i] = attenuation_lookup (old_img[i]);
    }
    vol->pix_type = PT_FLOAT;
    free (vol->img);
    vol->img = new_img;
}

void
drr_ray_trace_callback (
    void *callback_data, 
    int vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;

#if defined (DRR_PREPROCESS_ATTENUATION)
    cd->accum += vox_len * vox_value;
#if defined (DEBUG_INTENSITIES)
    printf ("len: %10g dens: %10g acc: %10g\n", 
	vox_len, vox_density, cd->accum);
#endif
#else
    accum += vox_len * attenuation_lookup (vox_value);
#endif
    cd->num_pix++;
}

double                            /* Return value: intensity of ray */
drr_ray_trace_exact (
    Volume *vol,                  /* Input: volume */
    Volume_limit *vol_limit,      /* Input: min/max coordinates of volume */
    double *p1in,                 /* Input: start point for ray */
    double *p2in                  /* Input: end point for ray */
)
{
    Callback_data cd;
    memset (&cd, 0, sizeof (Callback_data));

    ray_trace_exact (vol, vol_limit, &drr_ray_trace_callback, &cd, 
	p1in, p2in);
    return cd.accum;
}

void
drr_render_volume_perspective (
    Proj_image *proj,
    Volume *vol, 
    double ps[2], 
    char *multispectral_fn, 
    Drr_options *options
)
{
    double p1[3];
#if defined (VERBOSE)
    int rows = options->image_window[1] - options->image_window[0] + 1;
#endif
    int cols = options->image_window[3] - options->image_window[2] + 1;
    double ic_room[3];
    double ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double tmp[3];
    Volume_limit vol_limit;
    double nrm[3], pdn[3], prt[3];
    int r;
    Proj_matrix *pmat = proj->pmat;

    proj_matrix_get_nrm (pmat, nrm);
    proj_matrix_get_pdn (pmat, pdn);
    proj_matrix_get_prt (pmat, prt);

    /* Compute position of image center in room coordinates */
    vec3_scale3 (tmp, nrm, - pmat->sid);
    vec3_add3 (ic_room, pmat->cam, tmp);

    /* Compute incremental change in 3d position for each change 
       in panel row/column. */
    vec3_scale3 (incr_c, prt, ps[1]);
    vec3_scale3 (incr_r, pdn, ps[0]);

    /* Get position of upper left pixel on panel */
    vec3_copy (ul_room, ic_room);
    vec3_scale3 (tmp, incr_r, - pmat->ic[0]);
    vec3_add2 (ul_room, tmp);
    vec3_scale3 (tmp, incr_c, - pmat->ic[1]);
    vec3_add2 (ul_room, tmp);

    /* drr_ray_trace uses p1 & p2, p1 is the camera, p2 is in the 
       direction of the ray */
    vec3_copy (p1, pmat->cam);

#if defined (VERBOSE)
    printf ("NRM: %g %g %g\n", nrm[0], nrm[1], nrm[2]);
    printf ("PDN: %g %g %g\n", pdn[0], pdn[1], pdn[2]);
    printf ("PRT: %g %g %g\n", prt[0], prt[1], prt[2]);
    printf ("CAM: %g %g %g\n", pmat->cam[0], pmat->cam[1], pmat->cam[2]);
    printf ("ICR: %g %g %g\n", ic_room[0], ic_room[1], ic_room[2]);
    printf ("INCR_C: %g %g %g\n", incr_c[0], incr_c[1], incr_c[2]);
    printf ("INCR_R: %g %g %g\n", incr_r[0], incr_r[1], incr_r[2]);
    printf ("UL_ROOM: %g %g %g\n", ul_room[0], ul_room[1], ul_room[2]);
    printf ("IMG WDW: %d %d %d %d\n", 
	options->image_window[0], options->image_window[1], 
	options->image_window[2], options->image_window[3]);
#endif

    /* Compute volume boundary box */
    volume_limit_set (&vol_limit, vol);

    /* Compute the drr pixels */
#pragma omp parallel for
    for (r=options->image_window[0]; r<=options->image_window[1]; r++) {
	int c;
	double r_tgt[3];
	double tmp[3];
	double p2[3];

	//if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);
	vec3_copy (r_tgt, ul_room);
	vec3_scale3 (tmp, incr_r, (double) r);
	vec3_add2 (r_tgt, tmp);

	for (c=options->image_window[2]; c<=options->image_window[3]; c++) {
	    double value;
	    int idx = c - options->image_window[2] 
		+ (r - options->image_window[0]) * cols;

#if defined (VERBOSE)
	    printf ("Row: %4d/%d  Col:%4d/%d\n", r, rows, c, cols);
#endif
	    vec3_scale3 (tmp, incr_c, (double) c);
	    vec3_add3 (p2, r_tgt, tmp);

	    switch (options->interpolation) {
	    case INTERPOLATION_NONE:
		/* DRR_MSD is disabled */
		value = drr_ray_trace_exact (vol, &vol_limit, p1, p2);
		break;
	    case INTERPOLATION_TRILINEAR_EXACT:
		value = drr_trace_ray_trilin_exact (vol, p1, p2);
		break;
	    case INTERPOLATION_TRILINEAR_APPROX:
		value = drr_trace_ray_trilin_approx (vol, p1, p2);
		break;
	    }
	    value = value / 10;     /* Translate from pixels to cm*gm */
	    if (options->exponential_mapping) {
		value = exp(-value);
	    }
	    value = value * options->scale;   /* User requested scaling */

	    proj->img[idx] = (float) value;
	}
    }
}
