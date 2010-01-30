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
#include "readmha.h"
#include "timer.h"

//#define ULTRA_VERBOSE 1
//#define VERBOSE 1

typedef enum point_location Point_location;
enum point_location {
    POINTLOC_LEFT,
    POINTLOC_INSIDE,
    POINTLOC_RIGHT,
};

typedef struct volume_limit Volume_limit;
struct volume_limit {
    /* upper and lower limits of volume, including tolerances */
    double limits[2];

    /* dir == 0 if limits go from low to high */
    int dir;
};

/* According to NIST, the mass attenuation coefficient of H2O at 50 keV
   is 0.22 cm^2 per gram.  Thus, we scale by 0.022 per mm
   http://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html  */
float
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

float
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
init_multispectral (float* bins)
{
    memset (bins, 0, DRR_MSD_NUM_BINS*sizeof(float));
}

unsigned char
bin_multispectral (short pix_density)
{
    const short density_min = -800;
    const unsigned short pix_divisor = 1800 / DRR_MSD_NUM_BINS;

    pix_density -= density_min;
    if (pix_density < 0) return 0;
    pix_density /= pix_divisor;
    if (pix_density >= DRR_MSD_NUM_BINS) return DRR_MSD_NUM_BINS-1;
    return (unsigned char) pix_density;
}

void
accumulate_multispectral (float* bins, double pix_len, short pix_density)
{
    bins[bin_multispectral(pix_density)] += pix_len;
}

void
dump_multispectral (FILE* msd_fp, float* bins)
{
    fwrite (bins, sizeof(float), DRR_MSD_NUM_BINS, msd_fp);
}

double
drr_degeneracy_test (double* plane, double* ray)
{
    double dp = fabs(vec4_dot(plane, ray));
    return dp;
}

Point_location
drr_test_boundary (Volume_limit* vol_limit, double x)
{
    if (vol_limit->dir == 0) {
	if (x < vol_limit->limits[0]) {
	    return POINTLOC_LEFT;
	} else if (x > vol_limit->limits[1]) {
	    return POINTLOC_RIGHT;
	} else {
	    return POINTLOC_INSIDE;
	}
    } else {
	if (x > vol_limit->limits[0]) {
	    return POINTLOC_LEFT;
	} else if (x < vol_limit->limits[1]) {
	    return POINTLOC_RIGHT;
	} else {
	    return POINTLOC_INSIDE;
	}
    }
}

void
drr_trace_init_loopvars_nointerp (int* ai, int* aidir, double* ao, double* al, 
				  double pt, double ry, 
				  double offset, double samp)
{

    if (ry > 0) {
	*aidir = 1;
        *ai = (int) floor ((pt - offset + 0.5 * samp) / samp);
        *ao = samp - ((pt - offset + 0.5 * samp) - (*ai) * samp);
    } else {
	*aidir = -1;
        *ai = (int) floor ((pt - offset + 0.5 * samp) / samp);
        *ao = samp - ((*ai+1) * samp - (pt - offset + 0.5 * samp));
    }
    if (fabs(ry) > DRR_STRIDE_TOLERANCE) {
	*ao = *ao / fabs(ry);
	*al = samp / fabs(ry);
    } else {
	*ao = DRR_HUGE_DOUBLE;
	*al = DRR_HUGE_DOUBLE;
    }
}

/* Initialize loop variables.  Returns 1 if the segment intersects 
   the volume, and 0 if the segment does not intersect. */
int
drr_trace_init (int *ai_x,
		int *ai_y,
		int *ai_z,
		int *aixdir, 
		int *aiydir, 
		int *aizdir,
		double *ao_x, 
		double *ao_y, 
		double *ao_z,
		double *al_x, 
		double *al_y, 
		double *al_z,
		double *len,
		Volume* vol, 
		Volume_limit vol_limits[3], 
		double* p1, 
		double* p2 
		)
{
    int d;
    Point_location ploc[3][2];
    double alpha[3][2];
    double alpha_in, alpha_out;
    double ray[3];
    double ips[2][4];

    /* For each of 3 dimensions: x, y, and z */
    for (d = 0; d < 3; d++) {
	ploc[d][0] = drr_test_boundary (&vol_limits[d], p1[d]);
	ploc[d][1] = drr_test_boundary (&vol_limits[d], p2[d]);
	/* Reject segments which don't intersect the volume in this dimension */
	if (ploc[d][0] == POINTLOC_LEFT && ploc[d][1] == POINTLOC_LEFT) return 0;
	if (ploc[d][0] == POINTLOC_RIGHT && ploc[d][1] == POINTLOC_RIGHT) return 0;
    }

#if defined (ULTRA_VERBOSE)
    printf ("vol_limit[*][0] = %g %g %g\n", vol_limits[0].limits[0], vol_limits[1].limits[0], vol_limits[2].limits[0]);
    printf ("vol_limit[*][1] = %g %g %g\n", vol_limits[0].limits[1], vol_limits[1].limits[1], vol_limits[2].limits[1]);
    printf ("ploc[*][0]: %d %d %d\n", ploc[0][0], ploc[1][0], ploc[2][0]);
    printf ("ploc[*][1]: %d %d %d\n", ploc[0][1], ploc[1][1], ploc[2][1]);
#endif

    /* If we made it here, all three dimensions have some range of alpha
	where they intersects the volume.  However, these alphas might 
	not overlap.  We compute the alphas, then test overlapping 
	alphas to find the segment range within the volume.  */
    for (d = 0; d < 3; d++) {
	if (ploc[d][0] == POINTLOC_LEFT) {
	    alpha[d][0] = (vol_limits[d].limits[0] - p1[d]) / (p2[d] - p1[d]);
	} else if (ploc[d][0] == POINTLOC_RIGHT) {
	    alpha[d][0] = (p1[d] - vol_limits[d].limits[1]) / (p1[d] - p2[d]);
	} else {
	    alpha[d][0] = 0.0;
	}
	if (ploc[d][1] == POINTLOC_LEFT) {
	    alpha[d][1] = (vol_limits[d].limits[0] - p1[d]) / (p2[d] - p1[d]);
	} else if (ploc[d][1] == POINTLOC_RIGHT) {
	    alpha[d][1] = (p1[d] - vol_limits[d].limits[1]) / (p1[d] - p2[d]);
	} else {
	    alpha[d][1] = 1.0;
	}
    }

    /* alpha_in is the alpha where the segment enters the boundary, and 
       alpha_out is where it exits the boundary.  */
    alpha_in = alpha[0][0];
    alpha_out = alpha[0][1];
    for (d = 1; d < 3; d++) {
	if (alpha_in < alpha[d][0]) alpha_in = alpha[d][0];
	if (alpha_out > alpha[d][1]) alpha_out = alpha[d][1];
    }
#if defined (ULTRA_VERBOSE)
    printf ("alpha[*][0] = %g %g %g\n", alpha[0][0], alpha[1][0], alpha[2][0]);
    printf ("alpha[*][1] = %g %g %g\n", alpha[0][1], alpha[1][1], alpha[2][1]);
    printf ("alpha in/out = %g %g\n", alpha_in, alpha_out);
#endif

    /* If exit is before entrance, the segment does not intersect the volume */
    if (alpha_out - alpha_in < DRR_LEN_TOLERANCE) {
	return 0;
    }

    /* Create the volume intersection points */
    vec3_sub3 (ray, p2, p1);
    for (d = 0; d < 3; d++) {
	ips[0][d] = p1[d] + alpha_in * ray[d];
	ips[1][d] = p1[d] + alpha_out * ray[d];
    }
    vec3_normalize1 (ray);

#if defined (ULTRA_VERBOSE)
    printf ("ips[0] = %g %g %g\n", ips[0][0], ips[0][1], ips[0][2]);
    printf ("ips[1] = %g %g %g\n", ips[1][0], ips[1][1], ips[1][2]);
    printf ("ray = %g %g %g\n", ray[0], ray[1], ray[2]);
#endif

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x    // index of x
       aixdir  // x indices moving up or down?
       ao_x    // absolute length to next voxel crossing
       al_x    // length between voxel crossings
    */
    drr_trace_init_loopvars_nointerp (ai_x, aixdir, ao_x, al_x, 
	ips[0][0],
	ray[0], 
	vol->offset[0], 
	vol->pix_spacing[0]);
    drr_trace_init_loopvars_nointerp (ai_y, aiydir, ao_y, al_y, 
	ips[0][1],
	ray[1], 
	vol->offset[1], 
	vol->pix_spacing[1]);
    drr_trace_init_loopvars_nointerp (ai_z, aizdir, ao_z, al_z, 
	ips[0][2], 
	ray[2], 
	vol->offset[2], 
	vol->pix_spacing[2]);

#if defined (ULTRA_VERBOSE)
    printf ("aix = %d aixdir = %d aox = %g alx = %g\n", *ai_x, *aixdir, *ao_x, *al_x);
    printf ("aiy = %d aiydir = %d aoy = %g aly = %g\n", *ai_y, *aiydir, *ao_y, *al_y);
    printf ("aiz = %d aizdir = %d aoz = %g alz = %g\n", *ai_z, *aizdir, *ao_z, *al_z);
#endif

    *len = vec3_dist(&ips[0][0],&ips[1][0]);
    return 1;
}

double
drr_trace_ray_nointerp_2009 (Volume* vol, Volume_limit vol_limits[3], 
			     double* p1in, double* p2in, 
			     FILE* msd_fp)
{
    int ai_x, ai_y, ai_z;
    int aixdir, aiydir, aizdir;
    double ao_x, ao_y, ao_z;
    double al_x, al_y, al_z;
    double len;
    double aggr_len = 0.0;
    double accum = 0.0;
    int num_pix = 0;
    float msd_bins[DRR_MSD_NUM_BINS];
    float* img = (float*) vol->img;

#if defined (ULTRA_VERBOSE)
    printf ("p1in: %f %f %f\n", p1in[0], p1in[1], p1in[2]);
    printf ("p2in: %f %f %f\n", p2in[0], p2in[1], p2in[2]);
#endif

    if (!drr_trace_init (
		&ai_x,
		&ai_y,
		&ai_z,
		&aixdir, 
		&aiydir, 
		&aizdir,
		&ao_x, 
		&ao_y, 
		&ao_z,
		&al_x, 
		&al_y, 
		&al_z,
		&len,
		vol, vol_limits, p1in, p2in)) {
	return 0.0;
    }

    if (msd_fp) {
	init_multispectral (msd_bins);
    }

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x    // index of x
       aixdir  // x indices moving up or down?
       ao_x    // absolute length to next voxel crossing
       al_x    // length between voxel crossings
    */
    do {
	float* zz = (float*) &img[ai_z*vol->dim[0]*vol->dim[1]];
	float pix_density;
	double pix_len;
#if defined (ULTRA_VERBOSE)
	printf ("(%d %d %d) (%g,%g,%g)\n",ai_x,ai_y,ai_z,ao_x,ao_y,ao_z);
	printf ("aggr_len = %g, len = %g\n", aggr_len, len);
	fflush (stdout);
#endif
	pix_density = zz[ai_y*vol->dim[0]+ai_x];
	if ((ao_x < ao_y) && (ao_x < ao_z)) {
	    pix_len = ao_x;
	    aggr_len += ao_x;
	    ao_y -= ao_x;
	    ao_z -= ao_x;
	    ao_x = al_x;
	    ai_x += aixdir;
	} else if ((ao_y < ao_z)) {
	    pix_len = ao_y;
	    aggr_len += ao_y;
	    ao_x -= ao_y;
	    ao_z -= ao_y;
	    ao_y = al_y;
	    ai_y += aiydir;
	} else {
	    pix_len = ao_z;
	    aggr_len += ao_z;
	    ao_x -= ao_z;
	    ao_y -= ao_z;
	    ao_z = al_z;
	    ai_z += aizdir;
	}
#if defined (DRR_PREPROCESS_ATTENUATION)
	accum += pix_len * pix_density;
#if defined (DEBUG_INTENSITIES)
	printf ("len: %10g dens: %10g acc: %10g\n", 
		pix_len, pix_density, accum);
#endif
#else
	accum += pix_len * attenuation_lookup (pix_density);
#endif
	if (msd_fp) {
	    accumulate_multispectral (msd_bins, pix_len, pix_density);
	}
	num_pix++;
    } while (aggr_len+DRR_LEN_TOLERANCE < len);
    if (msd_fp) {
	dump_multispectral (msd_fp, msd_bins);
    }
    return accum;
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
    int d;
    double p1[3];
#if defined (ULTRA_VERBOSE)
    int rows = options->image_window[1] - options->image_window[0] + 1;
#endif
    int cols = options->image_window[3] - options->image_window[2] + 1;
    double ic_room[3];
    double ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double tmp[3];
    Volume_limit vol_limits[3];
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
    vec3_scale3 (incr_c, prt, ps[0]);
    vec3_scale3 (incr_r, pdn, ps[1]);

    /* Get position of upper left pixel on panel */
    vec3_copy (ul_room, ic_room);
    vec3_scale3 (tmp, incr_r, - pmat->ic[1]);
    vec3_add2 (ul_room, tmp);
    vec3_scale3 (tmp, incr_c, - pmat->ic[0]);
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
    for (d = 0; d < 3; d++) {
	vol_limits[d].limits[0] = vol->offset[d] - 0.5 * vol->pix_spacing[d];
	vol_limits[d].limits[1] = vol_limits[d].limits[0] 
	    + vol->dim[d] * vol->pix_spacing[d];
	if (vol_limits[d].limits[0] <= vol_limits[d].limits[1]) {
	    vol_limits[d].dir = 0;
	    vol_limits[d].limits[0] += DRR_BOUNDARY_TOLERANCE;
	    vol_limits[d].limits[1] -= DRR_BOUNDARY_TOLERANCE;
	} else {
	    vol_limits[d].dir = 1;
	    vol_limits[d].limits[0] -= DRR_BOUNDARY_TOLERANCE;
	    vol_limits[d].limits[1] += DRR_BOUNDARY_TOLERANCE;
	}
    }

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

#if defined (ULTRA_VERBOSE)
	    printf ("Row: %4d/%d  Col:%4d/%d\n", r, rows, c, cols);
#endif
	    vec3_scale3 (tmp, incr_c, (double) c);
	    vec3_add3 (p2, r_tgt, tmp);

	    switch (options->interpolation) {
	    case INTERPOLATION_NONE:
		/* DRR_MSD is disabled */
		value = drr_trace_ray_nointerp_2009 (vol, vol_limits, 
		    p1, p2, 0);
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
