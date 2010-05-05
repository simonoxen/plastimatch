/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math_util.h"
#include "proj_matrix.h"
#include "proton_dose.h"
#include "ray_trace_exact.h"
#include "ray_trace_uniform.h"
#include "volume.h"
#include "volume_limit.h"
#include "readmha.h"

//#define VERBOSE 1

typedef struct callback_data Callback_data;
struct callback_data {
    Volume *dose_vol;           /* Output image */
    Volume* depth_vol;		/* Radiographic depth volume */
    int* ires;			/* Aperture Dimensions */
    int ap_idx;			/* Current Aperture Coord */
    double accum;               /* Accumulated intensity */
};

static float
attenuation_lookup_weq (float density)
{
    const double min_hu = -1000.0;
    if (density <= min_hu) {
	return 0.0;
    } else {
	return ((density + 1000.0)/1000.0);
    }
}

static float
attenuation_lookup (float density)
{
    return attenuation_lookup_weq (density);
}

Volume*
create_depth_vol (
    int ct_dims[3],	// ct volume dimensions
    int ires[2],	// aperture dimensions
    float ray_step	// uniform ray step size
)
{
    int dv_dims[3];
    float dv_off[3] = {0.0f, 0.0f, 0.0f};
    float dv_ps[3] = {1.0f, 1.0f, 1.0f};
    float ct_diag;

    ct_diag =  ct_dims[0]*ct_dims[0];
    ct_diag += ct_dims[1]*ct_dims[1];
    ct_diag += ct_dims[2]*ct_dims[2];
    ct_diag = sqrt (ct_diag);

    dv_dims[0] = ires[0];	// rows = aperture rows
    dv_dims[1] = ires[1];	// cols = aperture cols
    dv_dims[2] = (int) floorf(ct_diag + 0.5) / ray_step;

    return volume_create(dv_dims, dv_off, dv_ps, PT_FLOAT, NULL, 0);
}

void
proton_dose_ray_trace_callback (
    void *callback_data, 
    int vox_index, 
    double vox_len, 
    float vox_value
)
{
    Callback_data *cd = (Callback_data *) callback_data;
    float *depth_img = (float*) cd->depth_vol->img;
    int ap_idx = cd->ap_idx;
    int ap_area = cd->ires[0] * cd->ires[1];
    int step_num = vox_index;

    cd->accum += vox_len * attenuation_lookup (vox_value);

    depth_img[ap_area*step_num + ap_idx] = cd->accum;
}

void
proton_dose_ray_trace (
    Volume *depth_vol,
    Volume *dose_vol,
    Volume *ct_vol,
    Volume_limit *vol_limit,
    double *p1, 
    double *p2, 
    int* ires,
    int ap_idx,
    Proton_dose_options *options
)
{
    Callback_data cd;
    double ray[3];
    double ip1[3];
    double ip2[3];

    vec3_sub3 (ray, p2, p1);
    vec3_normalize1 (ray);

    /* Test if ray intersects volume and create intersection points */
    if (!volume_limit_clip_ray (vol_limit, ip1, ip2, p1, ray)) {
	return;
    }

#if VERBOSE
    printf ("P1: %g %g %g\n", p1[0], p1[1], p1[2]);
    printf ("P2: %g %g %g\n", p2[0], p2[1], p2[2]);

    printf ("ip1 = %g %g %g\n", ip1[0], ip1[1], ip1[2]);
    printf ("ip2 = %g %g %g\n", ip2[0], ip2[1], ip2[2]);
    printf ("ray = %g %g %g\n", ray[0], ray[1], ray[2]);
#endif

    cd.accum = 0.0f;
    cd.ires = ires;
    cd.dose_vol = dose_vol;
    cd.depth_vol = depth_vol;
    cd.ap_idx = ap_idx;

    ray_trace_uniform (ct_vol,				// INPUT: CT volume
		vol_limit, 				// INPUT: CT volume bounding box
		&proton_dose_ray_trace_callback,	// INPUT: step action cbFunction
		&cd,					// INPUT: callback data
		ip1,					// INPUT: ray starting point
		ip2,					// INPUT: ray ending point
		options->ray_step);			// INPUT: uniform ray step size
}

void
proton_dose_compute (
    Volume *dose_vol,
    Volume *ct_vol,
    Proton_dose_options *options
)
{
    Volume* depth_vol;
    int ap_idx;

    int r;
    double p1[3];
    double ap_dist = 1000.;
    double nrm[3], pdn[3], prt[3], tmp[3];
    double ic_room[3];
    double ul_room[3];
    double incr_r[3];
    double incr_c[3];
    Volume_limit ct_limit;

    Proj_matrix *pmat;
    double cam[3] = { options->src[0], options->src[1], options->src[2] };
    double tgt[3] = { options->isocenter[0], options->isocenter[1], 
		      options->isocenter[2] };
    double vup[3] = { options->vup[0], options->vup[1], options->vup[2] };
    double ic[2] = { 4.5, 4.5 };
    double ps[2] = { 1., 1. };
    int ires[2] = { 10, 10 };

    pmat = proj_matrix_create ();
    proj_matrix_set (pmat, cam, tgt, vup, ap_dist, ic, ps, ires);

    proj_matrix_get_nrm (pmat, nrm);
    proj_matrix_get_pdn (pmat, pdn);
    proj_matrix_get_prt (pmat, prt);

    /* Compute position of aperture in room coordinates */
    vec3_scale3 (tmp, nrm, - pmat->sid);
    vec3_add3 (ic_room, pmat->cam, tmp);

    /* Compute incremental change in 3d position for each change 
       in aperture row/column. */
    vec3_scale3 (incr_c, prt, ps[1]);
    vec3_scale3 (incr_r, pdn, ps[0]);

    /* Get position of upper left pixel on panel */
    vec3_copy (ul_room, ic_room);
    vec3_scale3 (tmp, incr_r, - pmat->ic[0]);
    vec3_add2 (ul_room, tmp);
    vec3_scale3 (tmp, incr_c, - pmat->ic[1]);
    vec3_add2 (ul_room, tmp);

    /* drr_trace_ray uses p1 & p2, p1 is the camera, p2 is in the 
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
#endif

    /* Compute volume boundary box */
    volume_limit_set (&ct_limit, ct_vol);

    /* Create the depth volume */
    depth_vol = create_depth_vol (ct_vol->dim, ires, options->ray_step);

    //    ires[0] = ires[1] = 1;
    for (r = 0; r < ires[0]; r++) {
	int c;
	double r_tgt[3];
	double tmp[3];
	double p2[3];

	//if (r % 50 == 0) printf ("Row: %4d/%d\n", r, rows);
	vec3_copy (r_tgt, ul_room);
	vec3_scale3 (tmp, incr_r, (double) r);
	vec3_add2 (r_tgt, tmp);

	for (c = 0; c < ires[1]; c++) {
	    
	    vec3_scale3 (tmp, incr_c, (double) c);
	    vec3_add3 (p2, r_tgt, tmp);

	    ap_idx = c * ires[0] + r;
	    
	    proton_dose_ray_trace (depth_vol,
			    	dose_vol,
				ct_vol,
				&ct_limit,
				p1, p2,
				ires,
				ap_idx,
				options);
	}
    }

    if (options->debug) {
	    write_mha("depth_vol.mha", depth_vol);
    }
}
