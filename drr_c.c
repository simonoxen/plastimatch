/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"
#include "mathutil.h"
#include "drr_opts.h"
#include "readmha.h"

#define DRR_PLANE_RAY_TOLERANCE 1e-8
#define DRR_STRIDE_TOLERANCE 1e-10
#define DRR_HUGE_DOUBLE 1e10
#define DRR_LEN_TOLERANCE 1e-6
#define DRR_TOPLANE_TOLERANCE 1e-7

#define MSD_NUM_BINS 60

// #define ULTRA_VERBOSE 1
// #define VERBOSE 1

#define PREPROCESS_ATTENUATION 1
#define IMGTYPE float

#ifndef M_PI
#define M_PI            3.14159265358979323846
#endif
#ifndef M_TWOPI
#define M_TWOPI         (M_PI * 2.0)
#endif


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
preprocess_attenuation (Volume* vol)
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
    memset (bins, 0, MSD_NUM_BINS*sizeof(float));
}

unsigned char
bin_multispectral (short pix_density)
{
    const short density_min = -800;
    const unsigned short pix_divisor = 1800 / MSD_NUM_BINS;

    pix_density -= density_min;
    if (pix_density < 0) return 0;
    pix_density /= pix_divisor;
    if (pix_density >= MSD_NUM_BINS) return MSD_NUM_BINS-1;
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
    fwrite (bins, sizeof(float), MSD_NUM_BINS, msd_fp);
}

double
drr_degeneracy_test (double* plane, double* ray)
{
    double dp = fabs(vec4_dot(plane, ray));
    return dp;
}

void
drr_boundary_intersection_test (double ips[2][4], int *num_ips, double* plane, int* bdi, double* bdy, double* ray, double* p1, double* p2)
{
    double d1, d2;
    double ip[3];
    double sp[4];

    /* We already have two intersections */
    if (*num_ips == 2) {
	return;
    }

    /* No intersection when degenerate */
    if (drr_degeneracy_test(plane,ray) < DRR_PLANE_RAY_TOLERANCE) {
#if defined (VERBOSE)
	if (g_debug) printf ("Warning, degenerate ray\n");
	vec4_print_eol (stdout, plane);
	vec4_print_eol (stdout, ray);
#endif
	return;
    }

    /* Compute intersection of ray and plane into ip */
    d1 = vec4_dot (plane, p1);
    d2 = vec4_dot (plane, p2);

    vec3_copy(ip,p1);
    vec3_copy(sp,p2);
    vec3_sub2(sp,p1);
    vec3_scale2(sp,(d1/(d2-d1)));
    vec3_sub2(ip,sp);
    
#if defined (commentout)
    if (ip[bdi[0]]-DRR_TOPLANE_TOLERANCE < bdy[0] || ip[bdi[0]]+DRR_TOPLANE_TOLERANCE > bdy[1]) {
	return 0;
    }
    if (ip[bdi[1]]-DRR_TOPLANE_TOLERANCE < bdy[2] || ip[bdi[1]]+DRR_TOPLANE_TOLERANCE > bdy[3]) {
	return 0;
    }
#endif

    /* Compare intersection point against rectangular limits for this plane */
    if (ip[bdi[0]]+DRR_TOPLANE_TOLERANCE < bdy[0]) return;
    if (ip[bdi[0]]-DRR_TOPLANE_TOLERANCE > bdy[1]) return;
    if (ip[bdi[1]]+DRR_TOPLANE_TOLERANCE < bdy[2]) return;
    if (ip[bdi[1]]-DRR_TOPLANE_TOLERANCE > bdy[3]) return;

    /* Check corner case, where two planes intersect ray at same point */
    if (*num_ips == 1) {
	if (vec3_dist (ip, ips[0]) < DRR_TOPLANE_TOLERANCE) {
	    return;
	}
    }

    /* Good intersection, add to list */
    vec4_copy (ips[(*num_ips)++],ip);
}

/* Output is stored in ips (ips = intersection points) */
int
drr_compute_boundary_intersections (double ips[2][4], Volume* vol, 
				    double* p1, double* p2)
{
    /* Intersect ray with volume boundaries -- may not intersect in 
       degenerate cases 
       Boundaries are described by implicit eqn for plane:
       tau = [a b c d], where plane is (tau dot point) = 0
    */
    double ctx0[4] = {1,0,0,-(vol->xmin)};
    double ctx1[4] = {1,0,0,-(vol->xmax)};
    double cty0[4] = {0,1,0,-(vol->ymin)};
    double cty1[4] = {0,1,0,-(vol->ymax)};
    double ctz0[4] = {0,0,1,-(vol->zmin)};
    double ctz1[4] = {0,0,1,-(vol->zmax)};
    int ctx_bi[2] = {1,2};
    int cty_bi[2] = {0,2};
    int ctz_bi[2] = {0,1};
    double ctx_bd[4] = {vol->ymin,vol->ymax,vol->zmin,vol->zmax};
    double cty_bd[4] = {vol->xmin,vol->xmax,vol->zmin,vol->zmax};
    double ctz_bd[4] = {vol->xmin,vol->xmax,vol->ymin,vol->ymax};

    double rayh[4], p1h[4], p2h[4];
    int num_ips = 0;

    vec3_copy (p1h, p1);
    p1h[3] = 1.0;
    vec3_copy (p2h, p2);
    p2h[3] = 1.0;

    /* rayh = p2 - p1 */
    vec3_copy (rayh, p2);
    vec3_sub2 (rayh, p1);
    vec3_normalize1 (rayh);
    rayh[3] = 0.0;

    drr_boundary_intersection_test (ips, &num_ips, ctx0, ctx_bi, ctx_bd, rayh, p1h, p2h);
    drr_boundary_intersection_test (ips, &num_ips, ctx1, ctx_bi, ctx_bd, rayh, p1h, p2h);
    drr_boundary_intersection_test (ips, &num_ips, cty0, cty_bi, cty_bd, rayh, p1h, p2h);
    drr_boundary_intersection_test (ips, &num_ips, cty1, cty_bi, cty_bd, rayh, p1h, p2h);
    drr_boundary_intersection_test (ips, &num_ips, ctz0, ctz_bi, ctz_bd, rayh, p1h, p2h);
    drr_boundary_intersection_test (ips, &num_ips, ctz1, ctz_bi, ctz_bd, rayh, p1h, p2h);

    /* No intersection */
    if (num_ips < 2)
        return 0;
    else
	return 1;
}

/* Output is stored in ips (ips = intersection points) */
/* This version computes 1/2 a voxel within the volume, as used 
   for the interpolation methods */
int
drr_compute_boundary_intersections_2 (double ips[2][4], Volume* vol, 
				      double* p1, double* p2)
{
    /* Intersect ray with volume boundaries -- may not intersect in 
       degenerate cases 
       Boundaries are described by implicit eqn for plane:
       tau = [a b c d], where plane is (tau dot point) = 0
    */
    double ctx0[4] = {1,0,0,-(vol->xmin+vol->pix_spacing[0]/2.0)};
    double ctx1[4] = {1,0,0,-(vol->xmax-vol->pix_spacing[0]/2.0)};
    double cty0[4] = {0,1,0,-(vol->ymin+vol->pix_spacing[1]/2.0)};
    double cty1[4] = {0,1,0,-(vol->ymax-vol->pix_spacing[1]/2.0)};
    double ctz0[4] = {0,0,1,-(vol->zmin+vol->pix_spacing[2]/2.0)};
    double ctz1[4] = {0,0,1,-(vol->zmax-vol->pix_spacing[2]/2.0)};
    int ctx_bi[2] = {1,2};
    int cty_bi[2] = {0,2};
    int ctz_bi[2] = {0,1};
    double ctx_bd[4] = {vol->ymin,vol->ymax,vol->zmin,vol->zmax};
    double cty_bd[4] = {vol->xmin,vol->xmax,vol->zmin,vol->zmax};
    double ctz_bd[4] = {vol->xmin,vol->xmax,vol->ymin,vol->ymax};

    double rayh[4], p1h[4], p2h[4];
    int ipidx = 0;

    vec3_copy (p1h, p1);
    p1h[3] = 1.0;
    vec3_copy (p2h, p2);
    p2h[3] = 1.0;

    /* rayh = p2 - p1 */
    vec3_copy (rayh, p2);
    vec3_sub2 (rayh, p1);
    vec3_normalize1 (rayh);
    rayh[3] = 0.0;

#if defined (commentout)
    tr = drr_boundary_intersection_test (ip, ctx0, ctx_bi, ctx_bd, rayh, p1h, p2h);
    if (tr>0 && ipidx < 2) vec4_copy(ips[ipidx++],ip);
    tr = drr_boundary_intersection_test (ip, ctx1, ctx_bi, ctx_bd, rayh, p1h, p2h);
    if (tr>0 && ipidx < 2) vec4_copy(ips[ipidx++],ip);
    tr = drr_boundary_intersection_test (ip, cty0, cty_bi, cty_bd, rayh, p1h, p2h);
    if (tr>0 && ipidx < 2) vec4_copy(ips[ipidx++],ip);
    tr = drr_boundary_intersection_test (ip, cty1, cty_bi, cty_bd, rayh, p1h, p2h);
    if (tr>0 && ipidx < 2) vec4_copy(ips[ipidx++],ip);
    tr = drr_boundary_intersection_test (ip, ctz0, ctz_bi, ctz_bd, rayh, p1h, p2h);
    if (tr>0 && ipidx < 2) vec4_copy(ips[ipidx++],ip);
    tr = drr_boundary_intersection_test (ip, ctz1, ctz_bi, ctz_bd, rayh, p1h, p2h);
    if (tr>0 && ipidx < 2) vec4_copy(ips[ipidx++],ip);
#endif
    printf ("This function is unfinished");
    exit (-1);

    /* No intersection */
    if (ipidx < 2)
        return 0;
    else
	return 1;
}

void
drr_trace_init_loopvars_nointerp (int* ai, int* aidir, double* ao, double* al, 
				  double pt, double ry, double samp)
{
    if (ry > 0) {
	*aidir = 1;
        *ai = (int) floor((pt+DRR_TOPLANE_TOLERANCE) / samp);
        *ao = samp - (pt - (*ai) * samp);
    } else {
	*aidir = -1;
        *ai = (int) floor((pt-DRR_TOPLANE_TOLERANCE) / samp);
        *ao = samp - ((*ai+1) * samp - pt);
    }
    *al = samp;
    if (fabs(ry) > DRR_STRIDE_TOLERANCE) {
	*ao = *ao / fabs(ry);
	*al = *al / fabs(ry);
    } else {
	*ao = DRR_HUGE_DOUBLE;
	*al = DRR_HUGE_DOUBLE;
    }
}

void
drr_trace_init_loopvars_interp (int* ai, int* aidir, double* ao, double* al, 
				double pt, double ry, double samp)
{
    if (ry > 0) {
	*aidir = 1;
        *ai = (int) floor((pt+DRR_TOPLANE_TOLERANCE) / samp * 2.0);
	*ai = (*ai + 1) / 2;
        *ao = samp - (pt - (2*(*ai)-1) * samp / 2.0);
    } else {
	*aidir = -1;
        *ai = (int) floor((pt-DRR_TOPLANE_TOLERANCE) / samp * 2.0);
	*ai = (*ai + 1) / 2;
        *ao = samp - (((2*(*ai+1))-1) * samp / 2.0 - pt);
    }
    *al = samp;
    if (fabs(ry) > DRR_STRIDE_TOLERANCE) {
	*ao = *ao / fabs(ry);
	*al = *al / fabs(ry);
    } else {
	*ao = DRR_HUGE_DOUBLE;
	*al = DRR_HUGE_DOUBLE;
    }
}

void
drr_trace_init_loopvars_interp_2 (int* ai, int* aidir, double* ao, 
				  double* al, double pt, double ry, 
				  double samp)
{
    if (ry > 0) {
	*aidir = 1;
        *ai = (int) floor((pt-(samp/2.0)+DRR_TOPLANE_TOLERANCE) / samp);
        *ao = samp - ((pt+(samp/2.0)) - (*ai) * samp);
    } else {
	*aidir = -1;
        *ai = (int) floor((pt+(samp/2.0)-DRR_TOPLANE_TOLERANCE) / samp);
        *ao = samp - ((*ai+1) * samp - (pt-(samp/2.0)));
    }
    *al = samp;
    if (fabs(ry) > DRR_STRIDE_TOLERANCE) {
	*ao = *ao / fabs(ry);
	*al = *al / fabs(ry);
    } else {
	*ao = DRR_HUGE_DOUBLE;
	*al = DRR_HUGE_DOUBLE;
    }
}

double
drr_trace_ray_nointerp (Volume* vol, double* p1in, double* p2in, 
			FILE* msd_fp)
{
    double ips[2][4];
    double *p1 = &ips[0][0], *p2 = &ips[1][0];
    int ai_x, ai_y, ai_z;
    int aixdir, aiydir, aizdir;
    double ao_x, ao_y, ao_z;
    double al_x, al_y, al_z;
    double rayh[4];
    double len;
    double aggr_len = 0.0;
    double accum = 0.0;
    int num_pix = 0;
    float msd_bins[MSD_NUM_BINS];
    float* img = (float*) vol->img;

    if (!drr_compute_boundary_intersections (ips, vol, p1in, p2in)) {
	return 0.0;
    }
    len = vec3_dist(p1,p2);

#if defined (VERBOSE)
    printf ("p1i: ");
    vec3_print_eol (stdout,p1);
    printf ("p2i: ");
    vec3_print_eol (stdout,p2);
#endif

    vec3_sub3 (rayh, p2, p1);
    vec3_normalize1 (rayh);
    rayh[3] = 0.0;

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
    drr_trace_init_loopvars_nointerp (&ai_x, &aixdir, &ao_x, &al_x, 
				      p1[0] - vol->xmin, rayh[0], 
				      vol->pix_spacing[0]);
    drr_trace_init_loopvars_nointerp (&ai_y, &aiydir, &ao_y, &al_y, 
				      p1[1] - vol->ymin, rayh[1], 
				      vol->pix_spacing[1]);
    drr_trace_init_loopvars_nointerp (&ai_z, &aizdir, &ao_z, &al_z, 
				      p1[2] - vol->zmin, rayh[2], 
				      vol->pix_spacing[2]);
    do {
	float* zz = (float*) &img[ai_z*vol->dim[0]*vol->dim[1]];
	float pix_density;
	double pix_len;
#if defined (ULTRA_VERBOSE)
	printf ("(%d %d %d) (%g,%g,%g)\n",ai_x,ai_y,ai_z,ao_x,ao_y,ao_z);
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
#if defined (PREPROCESS_ATTENUATION)
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

double
interp_coefficient (double u0, double UU, double v0, double VV, 
		    double w0, double WW, int x, int y, int z)
{
    if (x == 1) {
	u0 = 1.0 - u0;
	UU = - UU;
    }
    if (y == 1) {
	v0 = 1.0 - v0;
	VV = - VV;
    }
    if (z == 1) {
	w0 = 1.0 - w0;
	WW = - WW;
    }
    return u0*v0*w0 
	    + (u0*v0*WW + u0*VV*w0 + UU*v0*w0) / 2.0
	    + (u0*VV*WW + UU*v0*WW + UU*VV*w0) / 3.0
	    + UU*VV*WW / 4.0;
}

/* GCS FIX: The last slice of voxels is wrong */
double
drr_trace_ray_trilin_exact (Volume* vol, double* p1in, double* p2in)
{
    double ips[2][4];
    double *p1 = &ips[0][0], *p2 = &ips[1][0];
    int ai_x, ai_y, ai_z;
    int aixdir, aiydir, aizdir;
    double ao_x, ao_y, ao_z;
    double ao_x0, ao_x1, ao_y0, ao_y1, ao_z0, ao_z1;
    double al_x, al_y, al_z;
    double rayh[4];
    double len;
    double aggr_len = 0.0;
    double accum = 0.0;
    int num_pix = 0;
    float* img = (float*) vol->img;

    if (!drr_compute_boundary_intersections (ips, vol, p1in, p2in)) {
	return 0.0;
    }
    len = vec3_dist(p1,p2);

#if defined (VERBOSE)
    printf ("p1i: ");
    vec3_print_eol (stdout,p1);
    printf ("p2i: ");
    vec3_print_eol (stdout,p2);
#endif

    vec3_sub3 (rayh, p2, p1);
    vec3_normalize1 (rayh);
    rayh[3] = 0.0;

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x    // index of x
       aixdir  // x indices moving up or down?
       ao_x    // absolute length to next voxel crossing
       al_x    // length between voxel crossings
    */
    drr_trace_init_loopvars_interp (&ai_x, &aixdir, &ao_x, &al_x, 
				    p1[0] - vol->xmin,
				    rayh[0], 
				    vol->pix_spacing[0]);
    drr_trace_init_loopvars_interp (&ai_y, &aiydir, &ao_y, &al_y, 
				    p1[1] - vol->ymin,
				    rayh[1], 
				    vol->pix_spacing[1]);
    drr_trace_init_loopvars_interp (&ai_z, &aizdir, &ao_z, &al_z, 
				    p1[2] - vol->zmin,
				    rayh[2], 
				    vol->pix_spacing[2]);
    do {
	int x1, x2, y1, y2, z1, z2;
	float pix111, pix112, pix121, pix122, 
		pix211, pix212, pix221, pix222;
	double pix_len;

	if (ai_x==0) {
	    x1 = ai_x;
	    x2 = ai_x;
	} else if (ai_x == vol->dim[2]) {
	    x1 = ai_x-1;
	    x2 = ai_x-1;
	} else {
	    x1 = ai_x-1;
	    x2 = ai_x;
	}
	if (ai_y==0) {
	    y1 = ai_y;
	    y2 = ai_y;
	} else if (ai_y == vol->dim[2]) {
	    y1 = ai_y-1;
	    y2 = ai_y-1;
	} else {
	    y1 = ai_y-1;
	    y2 = ai_y;
	}
	if (ai_z==0) {
	    z1 = ai_z;
	    z2 = ai_z;
	} else if (ai_z == vol->dim[2]) {
	    z1 = ai_z-1;
	    z2 = ai_z-1;
	} else {
	    z1 = ai_z-1;
	    z2 = ai_z;
	}

	pix111 = img[(z1*vol->dim[1]+y1)*vol->dim[0]+x1];
	pix112 = img[(z2*vol->dim[1]+y1)*vol->dim[0]+x1];
	pix121 = img[(z1*vol->dim[1]+y2)*vol->dim[0]+x1];
	pix122 = img[(z2*vol->dim[1]+y2)*vol->dim[0]+x1];
	pix211 = img[(z1*vol->dim[1]+y1)*vol->dim[0]+x2];
	pix212 = img[(z2*vol->dim[1]+y1)*vol->dim[0]+x2];
	pix221 = img[(z1*vol->dim[1]+y2)*vol->dim[0]+x2];
	pix222 = img[(z2*vol->dim[1]+y2)*vol->dim[0]+x2];

#if defined (ULTRA_VERBOSE)
	printf ("(%d %d %d) (%d %d %d) (%g,%g,%g) (%g,%g,%g)\n",
		ai_x,ai_y,ai_z,
		aixdir,aiydir,aizdir,
		ao_x,ao_y,ao_z,
		al_x,al_y,al_z);
#endif

	if (aixdir == +1) {
	    ao_x0 = al_x - ao_x;
	} else {
	    ao_x0 = ao_x;
	}
	if (aiydir == +1) {
	    ao_y0 = al_y - ao_y;
	} else {
	    ao_y0 = ao_y;
	}
	if (aizdir == +1) {
	    ao_z0 = al_z - ao_z;
	} else {
	    ao_z0 = ao_z;
	}

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
	    ao_z -= ao_y;
	    ao_x -= ao_y;
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
#if defined (commentout)
	if (aixdir == +1) {
	    ao_x1 = ao_x0 + pix_len;
	} else {
	    ao_x1 = ao_x0 - pix_len;
	}
	if (aiydir == +1) {
	    ao_y1 = ao_y0 + pix_len;
	} else {
	    ao_y1 = ao_y0 - pix_len;
	}
	if (aizdir == +1) {
	    ao_z1 = ao_z0 + pix_len;
	} else {
	    ao_z1 = ao_z0 - pix_len;
	}
#endif
	ao_x1 = ao_x0 + aixdir * pix_len;
	ao_y1 = ao_y0 + aiydir * pix_len;
	ao_z1 = ao_z0 + aizdir * pix_len;

#if defined (ULTRA_VERBOSE)
	printf ("AOXYZ = %g %g, %g %g, %g %g\n", ao_x0, ao_x1, 
		ao_y0, ao_y1, ao_z0, ao_z1);

#endif
	{
	    double u0 = ao_x0 / al_x;
	    double u1 = ao_x1 / al_x;
	    double UU = (u1 - u0);
	    double v0 = ao_y0 / al_y;
	    double v1 = ao_y1 / al_y;
	    double VV = (v1 - v0);
	    double w0 = ao_z0 / al_z;
	    double w1 = ao_z1 / al_z;
	    double WW = (w1 - w0);
//	    double tmp_accum;
#if defined (PREPROCESS_ATTENUATION)
	    accum += pix_len *
		    (interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1)
		     * pix111
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2)
		     * pix112
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1)
		     * pix121
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2)
		     * pix122
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1)
		     * pix211
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2)
		     * pix212
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1)
		     * pix221
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2)
		     * pix222);
#else
	    accum += pix_len *
		    (interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1)
		     * attenuation_lookup (pix111)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2)
		     * attenuation_lookup (pix112)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1)
		     * attenuation_lookup (pix121)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2)
		     * attenuation_lookup (pix122)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1)
		     * attenuation_lookup (pix211)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2)
		     * attenuation_lookup (pix212)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1)
		     * attenuation_lookup (pix221)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2)
		     * attenuation_lookup (pix222));
#endif
#if defined (ULTRA_VERBOSE)
	    printf ("UVW = (%g,%g,%g, * %g,%g,%g, * %g,%g,%g)\n", 
		    u0,u1,UU,v0,v1,VV,w0,w1,WW);
	    tmp_accum = interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2);
	    printf ("ACCUM = %g (%g,%g,%g,%g,%g,%g,%g,%g)\n", 
		    tmp_accum,
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2));
#endif
	    if (interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1) < 0 
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2) < 0) {
		printf ("Error with interp_coefficient()\n");
	    }
	}
	num_pix++;
    } while (aggr_len+DRR_LEN_TOLERANCE < len);
    return accum;
}


double
drr_trace_ray_trilin_exact_under_development (Volume* vol, double* p1in, 
					      double* p2in)
{
    double ips[2][4];
    double *p1 = &ips[0][0], *p2 = &ips[1][0];
    int ai_x, ai_y, ai_z;
    int aixdir, aiydir, aizdir;
    double ao_x, ao_y, ao_z;
//    double ao_x0, ao_x1, ao_y0, ao_y1, ao_z0, ao_z1;
    double al_x, al_y, al_z;
    double rayh[4];
    double len;
    double aggr_len = 0.0;
    double accum = 0.0;
    double ap_x0, ap_x1, ap_x, ap_y0, ap_y1, ap_y, ap_z0, ap_z1, ap_z;

    int num_pix = 0;
    float* img = (float*) vol->img;

    if (!drr_compute_boundary_intersections (ips, vol, p1in, p2in)) {
	return 0.0;
    }
    len = vec3_dist(p1,p2);

#if defined (VERBOSE)
    vec_set_fmt ("%12.10g ");
    printf ("p1i: ");
    vec3_print_eol (stdout,p1);
    printf ("p2i: ");
    vec3_print_eol (stdout,p2);
#endif

    if (!drr_compute_boundary_intersections_2 (ips, vol, p1in, p2in)) {
	return 0.0;
    }
    len = vec3_dist(p1,p2);

#if defined (VERBOSE)
    printf ("p1i: ");
    vec3_print_eol (stdout,p1);
    printf ("p2i: ");
    vec3_print_eol (stdout,p2);
#endif

    vec3_sub3 (rayh, p2, p1);
    vec3_normalize1 (rayh);
    rayh[3] = 0.0;

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x    // index of x
       aixdir  // x indices moving up or down?
       ao_x    // absolute length to next voxel crossing
       al_x    // length between voxel crossings
    */
    drr_trace_init_loopvars_interp_2 (&ai_x, &aixdir, &ao_x, &al_x, 
				      p1[0] - vol->xmin,
				      rayh[0], 
				      vol->pix_spacing[0]);
    drr_trace_init_loopvars_interp_2 (&ai_y, &aiydir, &ao_y, &al_y, 
				      p1[1] - vol->ymin,
				      rayh[1], 
				      vol->pix_spacing[1]);
    drr_trace_init_loopvars_interp_2 (&ai_z, &aizdir, &ao_z, &al_z, 
				      p1[2] - vol->zmin,
				      rayh[2], 
				      vol->pix_spacing[2]);

    /* 
       ap_x     // intra-voxel position of x
       ap_x0    // intra-voxel position of x after a crossing
     */
    if (aixdir == +1) {
	ap_x0 = 0.0;
	ap_x1 = 1.0;
	ap_x = ao_x;
    } else {
	ap_x0 = 1.0;
	ap_x1 = 0.0;
	ap_x = al_x - ao_x;
    }
    if (aiydir == +1) {
	ap_y0 = 0.0;
	ap_y1 = 1.0;
	ap_y = ao_y;
    } else {
	ap_y0 = 1.0;
	ap_y1 = 0.0;
	ap_y = al_y - ao_y;
    }
    if (aizdir == +1) {
	ap_z0 = 0.0;
	ap_z1 = 1.0;
	ap_z = ao_z;
    } else {
	ap_z0 = 1.0;
	ap_z1 = 0.0;
	ap_z = al_z - ao_z;
    }
    do {
	int x1, x2, y1, y2, z1, z2;
	float pix111, pix112, pix121, pix122, 
		pix211, pix212, pix221, pix222;
	double pix_len;

#if defined (ULTRA_VERBOSE)
	printf ("(%d %d %d) (%d %d %d) (%g,%g,%g) (%g,%g,%g)\n",
		ai_x,ai_y,ai_z,
		aixdir,aiydir,aizdir,
		ao_x,ao_y,ao_z,
		al_x,al_y,al_z);
#endif

	double ap_x_init = ap_x;
	double ap_y_init = ap_y;
	double ap_z_init = ap_z;
	double ap_x_end;
	double ap_y_end;
	double ap_z_end;

	x1 = ai_x;
	x2 = ai_x + 1;
	y1 = ai_y;
	y2 = ai_y + 1;
	z1 = ai_z;
	z2 = ai_z + 1;

	pix111 = img[(z1*vol->dim[1]+y1)*vol->dim[0]+x1];
	pix112 = img[(z2*vol->dim[1]+y1)*vol->dim[0]+x1];
	pix121 = img[(z1*vol->dim[1]+y2)*vol->dim[0]+x1];
	pix122 = img[(z2*vol->dim[1]+y2)*vol->dim[0]+x1];
	pix211 = img[(z1*vol->dim[1]+y1)*vol->dim[0]+x2];
	pix212 = img[(z2*vol->dim[1]+y1)*vol->dim[0]+x2];
	pix221 = img[(z1*vol->dim[1]+y2)*vol->dim[0]+x2];
	pix222 = img[(z2*vol->dim[1]+y2)*vol->dim[0]+x2];

	if ((ao_x < ao_y) && (ao_x < ao_z)) {
	    pix_len = ao_x;
	    aggr_len += ao_x;
	    ao_y -= ao_x;
	    ao_z -= ao_x;
	    ao_x = al_x;
	    ai_x += aixdir;
	    ap_x = ap_x0;
	    ap_y += aiydir * ao_x / al_y;
	    ap_z += aizdir * ao_x / al_z;
	    ap_x_end = ap_x1;
	    ap_y_end = ap_y;
	    ap_z_end = ap_z;
	} else if ((ao_y < ao_z)) {
	    pix_len = ao_y;
	    aggr_len += ao_y;
	    ao_z -= ao_y;
	    ao_x -= ao_y;
	    ao_y = al_y;
	    ap_y = ap_y0;
	    ai_y += aiydir;
	    ap_x += aixdir * ao_y / al_x;
	    ap_y = ap_y0;
	    ap_z += aizdir * ao_y / al_z;
	    ap_x_end = ap_x;
	    ap_y_end = ap_y1;
	    ap_z_end = ap_z;
	} else {
	    pix_len = ao_z;
	    aggr_len += ao_z;
	    ao_x -= ao_z;
	    ao_y -= ao_z;
	    ao_z = al_z;
	    ap_z = ap_z0;
	    ai_z += aizdir;
	    ap_x += aixdir * ao_z / al_x;
	    ap_y += aiydir * ao_z / al_y;
	    ap_z = ap_z0;
	    ap_x_end = ap_x;
	    ap_y_end = ap_y;
	    ap_z_end = ap_z1;
	}

#if defined (ULTRA_VERBOSE)
	printf ("LEN = %g AGG = %g\n", len, aggr_len);
	printf ("AOXYZ = %g %g, %g %g, %g %g\n", 
		ap_x_init, ap_x_end, 
		ap_y_init, ap_y_end, 
		ap_z_init, ap_z_end);
#endif
	{
	    double u0 = ap_x_init;
	    double UU = ap_x_end - ap_x_init;
	    double v0 = ap_y_init;
	    double VV = ap_y_end - ap_y_init;
	    double w0 = ap_z_init;
	    double WW = ap_y_end - ap_y_init;
//	    double tmp_accum;
#if defined (PREPROCESS_ATTENUATION)
	    accum += pix_len *
		    (interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1)
		     * pix111
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2)
		     * pix112
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1)
		     * pix121
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2)
		     * pix122
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1)
		     * pix211
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2)
		     * pix212
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1)
		     * pix221
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2)
		     * pix222);
#else
	    accum += pix_len *
		    (interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1)
		     * attenuation_lookup (pix111)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2)
		     * attenuation_lookup (pix112)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1)
		     * attenuation_lookup (pix121)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2)
		     * attenuation_lookup (pix122)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1)
		     * attenuation_lookup (pix211)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2)
		     * attenuation_lookup (pix212)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1)
		     * attenuation_lookup (pix221)
		     + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2)
		     * attenuation_lookup (pix222));
#endif
#if defined (ULTRA_VERBOSE)
	    printf ("UVW = (%g,%g, * %g,%g * %g,%g)\n", 
		    u0,UU,v0,VV,w0,WW);
	    tmp_accum = interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1)
		    + interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2);
	    printf ("ACCUM = %g (%g,%g,%g,%g,%g,%g,%g,%g)\n", 
		    tmp_accum,
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1),
		    interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2));
	    if (interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,1) < 0 
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,1,1,2) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,1) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,1,2,2) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,1) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,1,2) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,1) < 0
		|| interp_coefficient (u0,UU,v0,VV,w0,WW,2,2,2) < 0) {
		printf ("Error with interp_coefficient()\n");
	    }
#endif
	}
	num_pix++;
    } while (aggr_len+DRR_LEN_TOLERANCE < len);
    return accum;
}

double
approx_coefficient (double UU, double VV, double WW, int x, int y, int z)
{
    if (x == 1) {
	UU = 1.0 - UU;
    }
    if (y == 1) {
	VV = 1.0 - VV;
    }
    if (z == 1) {
	WW = 1.0 - WW;
    }
    return UU * VV * WW;
}

double
drr_trace_ray_trilin_approx (Volume* vol, double* p1in, double* p2in)
{
    double ips[2][4];
    double *p1 = &ips[0][0], *p2 = &ips[1][0];
    int ai_x, ai_y, ai_z;
    int aixdir, aiydir, aizdir;
    double ao_x, ao_y, ao_z;
    double ao_x0, ao_x1, ao_y0, ao_y1, ao_z0, ao_z1;
    double al_x, al_y, al_z;
    double rayh[4];
    double len;
    double aggr_len = 0.0;
    double accum = 0.0;
    int num_pix = 0;
    float* img = (float*) vol->img;

    if (!drr_compute_boundary_intersections (ips, vol, p1in, p2in)) {
	return 0.0;
    }
    len = vec3_dist(p1,p2);

#if defined (VERBOSE)
    printf ("p1i: ");
    vec3_print_eol (stdout,p1);
    printf ("p2i: ");
    vec3_print_eol (stdout,p2);
#endif

    vec3_sub3 (rayh, p2, p1);
    vec3_normalize1 (rayh);
    rayh[3] = 0.0;

    /* We'll go from p1 to p2 */
    /* Variable notation:
       ai_x    // index of x
       aixdir  // x indices moving up or down?
       ao_x    // absolute length to next voxel crossing
       al_x    // length between voxel crossings
    */
    drr_trace_init_loopvars_interp (&ai_x, &aixdir, &ao_x, &al_x, 
				    p1[0] - vol->xmin, rayh[0], 
				    vol->pix_spacing[0]);
    drr_trace_init_loopvars_interp (&ai_y, &aiydir, &ao_y, &al_y, 
				    p1[1] - vol->ymin, rayh[1], 
				    vol->pix_spacing[1]);
    drr_trace_init_loopvars_interp (&ai_z, &aizdir, &ao_z, &al_z, 
				    p1[2] - vol->zmin, rayh[2], 
				    vol->pix_spacing[2]);
    do {
	int x1, x2, y1, y2, z1, z2;
	float pix111, pix112, pix121, pix122, 
		pix211, pix212, pix221, pix222;
	double pix_len;

	if (ai_x==0) {
	    x1 = ai_x;
	    x2 = ai_x;
	} else if (ai_x == vol->dim[2]) {
	    x1 = ai_x-1;
	    x2 = ai_x-1;
	} else {
	    x1 = ai_x-1;
	    x2 = ai_x;
	}
	if (ai_y==0) {
	    y1 = ai_y;
	    y2 = ai_y;
	} else if (ai_y == vol->dim[2]) {
	    y1 = ai_y-1;
	    y2 = ai_y-1;
	} else {
	    y1 = ai_y-1;
	    y2 = ai_y;
	}
	if (ai_z==0) {
	    z1 = ai_z;
	    z2 = ai_z;
	} else if (ai_z == vol->dim[2]) {
	    z1 = ai_z-1;
	    z2 = ai_z-1;
	} else {
	    z1 = ai_z-1;
	    z2 = ai_z;
	}
	pix111 = img[(z1*vol->dim[1]+y1)*vol->dim[0]+x1];
	pix112 = img[(z2*vol->dim[1]+y1)*vol->dim[0]+x1];
	pix121 = img[(z1*vol->dim[1]+y2)*vol->dim[0]+x1];
	pix122 = img[(z2*vol->dim[1]+y2)*vol->dim[0]+x1];
	pix211 = img[(z1*vol->dim[1]+y1)*vol->dim[0]+x2];
	pix212 = img[(z2*vol->dim[1]+y1)*vol->dim[0]+x2];
	pix221 = img[(z1*vol->dim[1]+y2)*vol->dim[0]+x2];
	pix222 = img[(z2*vol->dim[1]+y2)*vol->dim[0]+x2];
#if defined (ULTRA_VERBOSE)
	printf ("(%d %d %d) (%d %d %d) (%g,%g,%g) (%g,%g,%g)\n",
		ai_x,ai_y,ai_z,
		aixdir,aiydir,aizdir,
		ao_x,ao_y,ao_z,
		al_x,al_y,al_z);
#endif
	if (aixdir == +1) {
	    ao_x0 = al_x - ao_x;
	} else {
	    ao_x0 = ao_x;
	}
	if (aiydir == +1) {
	    ao_y0 = al_y - ao_y;
	} else {
	    ao_y0 = ao_y;
	}
	if (aizdir == +1) {
	    ao_z0 = al_z - ao_z;
	} else {
	    ao_z0 = ao_z;
	}

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
	    ao_z -= ao_y;
	    ao_x -= ao_y;
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

	if (aixdir == +1) {
	    ao_x1 = ao_x0 + pix_len;
	} else {
	    ao_x1 = ao_x0 - pix_len;
	}
	if (aiydir == +1) {
	    ao_y1 = ao_y0 + pix_len;
	} else {
	    ao_y1 = ao_y0 - pix_len;
	}
	if (aizdir == +1) {
	    ao_z1 = ao_z0 + pix_len;
	} else {
	    ao_z1 = ao_z0 - pix_len;
	}
	
#if defined (ULTRA_VERBOSE)
	printf ("AOXYZ = %g %g, %g %g, %g %g\n", ao_x0, ao_x1, 
		ao_y0, ao_y1, ao_z0, ao_z1);

#endif
	{
	    double u0 = ao_x0 / al_x;
	    double u1 = ao_x1 / al_x;
	    double UU = (u1 + u0) / 2.0;
	    double v0 = ao_y0 / al_y;
	    double v1 = ao_y1 / al_y;
	    double VV = (v1 + v0) / 2.0;
	    double w0 = ao_z0 / al_z;
	    double w1 = ao_z1 / al_z;
	    double WW = (w1 + w0) / 2.0;
	    //	    double tmp_accum;
#if defined (PREPROCESS_ATTENUATION)
	    accum += pix_len * 
		    (approx_coefficient (UU,VV,WW,1,1,1)
		     * pix111
		     + approx_coefficient (UU,VV,WW,1,1,2)
		     * pix112
		     + approx_coefficient (UU,VV,WW,1,2,1)
		     * pix121
		     + approx_coefficient (UU,VV,WW,1,2,2)
		     * pix122
		     + approx_coefficient (UU,VV,WW,2,1,1)
		     * pix211
		     + approx_coefficient (UU,VV,WW,2,1,2)
		     * pix212
		     + approx_coefficient (UU,VV,WW,2,2,1)
		     * pix221
		     + approx_coefficient (UU,VV,WW,2,2,2)
		     * pix222);
#else
	    accum += pix_len * 
		    (approx_coefficient (UU,VV,WW,1,1,1)
		     * attenuation_lookup (pix111)
		     + approx_coefficient (UU,VV,WW,1,1,2)
		     * attenuation_lookup (pix112)
		     + approx_coefficient (UU,VV,WW,1,2,1)
		     * attenuation_lookup (pix121)
		     + approx_coefficient (UU,VV,WW,1,2,2)
		     * attenuation_lookup (pix122)
		     + approx_coefficient (UU,VV,WW,2,1,1)
		     * attenuation_lookup (pix211)
		     + approx_coefficient (UU,VV,WW,2,1,2)
		     * attenuation_lookup (pix212)
		     + approx_coefficient (UU,VV,WW,2,2,1)
		     * attenuation_lookup (pix221)
		     + approx_coefficient (UU,VV,WW,2,2,2)
		     * attenuation_lookup (pix222));
#endif

#if defined (ULTRA_VERBOSE)
	    printf ("UVW = (%g,%g,%g, * %g,%g,%g, * %g,%g,%g)\n", 
		    u0,u1,UU,v0,v1,VV,w0,w1,WW);
	    tmp_accum = approx_coefficient (UU,VV,WW,1,1,1)
		    + approx_coefficient (UU,VV,WW,1,1,2)
		    + approx_coefficient (UU,VV,WW,1,2,1)
		    + approx_coefficient (UU,VV,WW,1,2,2)
		    + approx_coefficient (UU,VV,WW,2,1,1)
		    + approx_coefficient (UU,VV,WW,2,1,2)
		    + approx_coefficient (UU,VV,WW,2,2,1)
		    + approx_coefficient (UU,VV,WW,2,2,2);
	    printf ("ACCUM = %g (%g,%g,%g,%g,%g,%g,%g,%g)\n", 
		    tmp_accum,
		    approx_coefficient (UU,VV,WW,1,1,1),
		    approx_coefficient (UU,VV,WW,1,1,2),
		    approx_coefficient (UU,VV,WW,1,2,1),
		    approx_coefficient (UU,VV,WW,1,2,2),
		    approx_coefficient (UU,VV,WW,2,1,1),
		    approx_coefficient (UU,VV,WW,2,1,2),
		    approx_coefficient (UU,VV,WW,2,2,1),
		    approx_coefficient (UU,VV,WW,2,2,2));
	    if (approx_coefficient (UU,VV,WW,1,1,1) < 0 
		|| approx_coefficient (UU,VV,WW,1,1,2) < 0
		|| approx_coefficient (UU,VV,WW,1,2,1) < 0
		|| approx_coefficient (UU,VV,WW,1,2,2) < 0
		|| approx_coefficient (UU,VV,WW,2,1,1) < 0
		|| approx_coefficient (UU,VV,WW,2,1,2) < 0
		|| approx_coefficient (UU,VV,WW,2,2,1) < 0
		|| approx_coefficient (UU,VV,WW,2,2,2) < 0) {
		printf ("Error with approx_coefficient()\n");
	    }
#endif
	}
	num_pix++;
    } while (aggr_len+DRR_LEN_TOLERANCE < len);
    return accum;
}

void
drr_render_volume_orthographic (Volume* volume)
{
}

void
drr_write_projection_matrix (Volume* vol, double* cam, 
			     double* tgt, double* vup,
			     double sid, double* ic,
			     double* ps, int* ires,
			     char* out_fn)
{
    double extrinsic[16];
    double intrinsic[12];
    double projection[12];
    const int cols = 4;
    double sad;

    double nrm[3];
    double vrt[3];
    double vup_tmp[3];  /* Don't overwrite vup */

    FILE* fp;

    vec_zero (extrinsic, 16);
    vec_zero (intrinsic, 12);

    /* Compute image coordinate sys (nrm,vup,vrt) relative to room coords.
       ---------------
       nrm = tgt - cam
       vrt = nrm x vup
       vup = vrt x nrm
       ---------------
    */
    vec3_sub3 (nrm, tgt, cam);
    vec3_normalize1 (nrm);
    vec3_cross (vrt, nrm, vup);
    vec3_normalize1 (vrt);
    vec3_cross (vup_tmp, vrt, nrm);
    vec3_normalize1 (vup_tmp);

    /* !!! But change nrm here to -nrm */
    vec3_scale2 (nrm, -1.0);

    /* Build extrinsic matrix */
    vec3_copy (&extrinsic[0], vrt);
    vec3_copy (&extrinsic[4], vup_tmp);
    vec3_copy (&extrinsic[8], nrm);
    sad = vec3_len (cam);
    m_idx(extrinsic,cols,2,3) = - sad;
    m_idx(extrinsic,cols,3,3) = 1.0;

    /* Build intrinsic matrix */
    m_idx(intrinsic,cols,0,1) = - 1 / ps[0];
    m_idx(intrinsic,cols,1,0) = 1 / ps[1];
    m_idx(intrinsic,cols,2,2) = - 1 / sid;
    //    m_idx(intrinsic,cols,0,3) = ic[0];
    //    m_idx(intrinsic,cols,1,3) = ic[1];

    mat_mult_mat (projection, intrinsic,3,4, extrinsic,4,4);

#if defined (VERBOSE)
    printf ("Extrinsic:\n");
    matrix_print_eol (stdout, extrinsic, 4, 4);
    printf ("Intrinsic:\n");
    matrix_print_eol (stdout, intrinsic, 3, 4);
    printf ("Projection:\n");
    matrix_print_eol (stdout, projection, 3, 4);
#endif

    fp = fopen (out_fn, "w");
    if (!fp) {
	fprintf (stderr, "Error opening %s for write\n", out_fn);
	exit (-1);
    }
    fprintf (fp, "%18.8e %18.8e\n", ic[0], ic[1]);
    fprintf (fp,
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n" 
	"%18.8e %18.8e %18.8e %18.8e\n", 
	projection[0], projection[1], projection[2], projection[3],
	projection[4], projection[5], projection[6], projection[7],
	projection[8], projection[9], projection[10], projection[11]
	);
    fprintf (fp, "%18.8e\n%18.8e\n", sad, sid);
    fprintf (fp, "%18.8e %18.8e %18.8e\n", nrm[0], nrm[1], nrm[2]);
    fclose (fp);
}

void
drr_render_volume_perspective (Volume* vol, double* cam, 
			       double* tgt, double* vup,
			       double sid, double* ic,
			       double* ps, int* ires,
			       char* image_fn, 
			       char* multispectral_fn, 
			       MGHDRR_Options* options)
{
    double nrm[3];
    double vrt[3];
    double p1[3], p2[3];
    //    int res_r = ires[0];
    //    int res_c = ires[1];
    int res_r = options->image_window[1] - options->image_window[0] + 1;
    int res_c = options->image_window[3] - options->image_window[2] + 1;
    double pp_r = - ps[0];
    double pp_c = ps[1];
    double ic_room[3];
    double ul_room[3];
    double incr_r[3];
    double incr_c[3];
    double r_tgt[3];
    double tmp[3];

    FILE *pgm_fp, *msd_fp;
    int r, c;
    double value;

    /* Compute image coordinate sys (nrm,vup,vrt) relative to room coords.
       ---------------
       nrm = tgt - cam
       vrt = nrm x vup
       vup = vrt x nrm
       ---------------
    */
    vec3_sub3 (nrm, tgt, cam);
    vec3_normalize1 (nrm);
    vec3_cross (vrt, nrm, vup);
    vec3_normalize1 (vrt);
    vec3_cross (vup, vrt, nrm);
    vec3_normalize1 (vup);

    /* Compute position of image center in room coordinates */
    vec3_scale3 (tmp, nrm, sid);
    vec3_add3 (ic_room, cam, tmp);
#if defined (VERBOSE)
    printf ("icr: ");
    vec3_print_eol (stdout, ic_room);
#endif

    /* Compute base and increments for pixels. Base is at upper left, 
       and increment with increasing rows and columns. */
    vec3_scale3 (incr_r, vup, pp_r);
    vec3_scale3 (incr_c, vrt, pp_c);
    vec3_copy (ul_room, ic_room);
    vec3_scale3 (tmp, incr_r, -ic[0]);
    vec3_add2 (ul_room, tmp);
    vec3_scale3 (tmp, incr_c, -ic[1]);
    vec3_add2 (ul_room, tmp);

    /* drr_trace_ray uses p1 & p2, p1 is the camera, p2 is in the 
       direction of the ray */
    vec3_copy (p1, cam);

    pgm_fp = fopen(image_fn,"wb");
    if (!pgm_fp) {
	fprintf (stderr, "Error opening %s for write\n", image_fn);
	exit (-1);
    }
    if (options->multispectral) {
	msd_fp = fopen(multispectral_fn,"wb");
	if (!msd_fp) {
	    fprintf (stderr, "Error opening %s for write\n", multispectral_fn);
	    exit (-1);
	}
    } else {
	msd_fp = 0;
    }

    if (!strcmp(&options->output_format,"pgm")) {
	fprintf (pgm_fp, 
		 "P2\n"
		 "# Created by mghdrr\n"
		 "%d %d\n"
		 "65536\n",
		 res_c, res_r);
    } else if (!strcmp(&options->output_format,"pfm")){
	fprintf (pgm_fp, 
		 "Pf\n"
		 "%d %d\n"
		 "-1\n",
		 res_c, res_r);
    } 
    //else{
    //	fprintf(pgm_fp, "%d\n");
    //}


    for (r=options->image_window[0]; r<=options->image_window[1]; r++) {
	if (r % 50 == 0) printf ("Row: %4d/%d\n",r,res_r);
	vec3_copy (r_tgt, ul_room);
	vec3_scale3 (tmp, incr_r, (double) r);
	vec3_add2 (r_tgt, tmp);
	for (c=options->image_window[2]; c<=options->image_window[3]; c++) {
	    vec3_scale3 (tmp, incr_c, (double) c);
	    vec3_add3 (p2, r_tgt, tmp);

	    switch (options->interpolation) {
	    case INTERPOLATION_NONE:
		value = drr_trace_ray_nointerp (vol,p1,p2,msd_fp);
		break;
	    case INTERPOLATION_TRILINEAR_EXACT:
		value = drr_trace_ray_trilin_exact (vol,p1,p2);
		break;
	    case INTERPOLATION_TRILINEAR_APPROX:
		value = drr_trace_ray_trilin_approx (vol,p1,p2);
		break;
	    }
	    value = value / 10;     /* Translate from pixels to cm*gm */
	    if (options->exponential_mapping) {
		value = exp(-value);
	    }
	    value = value * options->scale;   /* User requested scaling */
	    if (!strcmp(&options->output_format,"pgm")){
		if (options->exponential_mapping) {
		    value = value * 65536;
		} else {
		    value = value * 15000;
		}
		if (value > 65536) {
		    value = 65536;
		} else if (value < 0) {
		    value = 0;
		}
		fprintf (pgm_fp,"%d ", ROUND_INT(value));
	    } else if (!strcmp(&options->output_format,"pfm")){
		float fv = (float) value;
		fwrite (&fv, sizeof(float), 1, pgm_fp);
		//fprintf (pgm_fp,"%g ",value);
	    } else {
		short fv = (short) value;
		fwrite (&fv, sizeof(short), 1, pgm_fp);
	    }
        }
	if (!strcmp(&options->output_format,"pgm")) {
	    fprintf (pgm_fp,"\n");
	}
    }
    fclose (pgm_fp);
    printf ("done.\n");
}

/* All distances in mm */
void
drr_render_volumes (Volume* vol, MGHDRR_Options* options)

{
    int a;

    //    double cam_ap[3] = {0.0, -1.0, 0.0};
    //    double cam_lat[3] = {-1.0, 0.0, 0.0};
    //    double* cam = cam_ap;
    //    double* cam = cam_lat;

    double vup[3] = {0, 0, 1};
    double tgt[3] = {0.0, 0.0, 0.0};
    double nrm[3];
    double tmp[3];

    /* Set source-to-axis distance */
    double sad = options->sad;

    /* Set source-to-image distance */
    double sid = options->sid;

    /* Set image resolution */
    int ires[2] = { options->image_resolution[0],
		    options->image_resolution[1] };

    /* Set physical size of imager in mm */
    //    int isize[2] = { 300, 400 };      /* Actual resolution */
    int isize[2] = { options->image_size[0],
		     options->image_size[1] };

    /* Set ic = image center (in pixels), and ps = pixel size (in mm)
       Note: pixels are numbered from 0 to ires-1 */
    double ic[2] = { options->image_center[0],
		     options->image_center[1] };
    
    /* Set pixel size in mm */
    double ps[2] = { (double)isize[0]/(double)ires[0], 
		     (double)isize[1]/(double)ires[1] };

    /* Loop through camera angles */
    for (a = 0; a < options->num_angles; a++) {
	double angle = a * options->angle_diff;
	double cam[3];
	char out_fn[256];
	char multispectral_fn[256];

	cam[0] = cos(angle);
	cam[1] = sin(angle);
	cam[2] = 0.0;
	
	printf ("Rendering DRR %d\n", a);

	/* Place camera at distance "sad" from the volume isocenter */
	vec3_sub3 (nrm, tgt, cam);
	vec3_normalize1 (nrm);
	vec3_scale3 (tmp, nrm, sad);
	vec3_copy (cam, tgt);
	vec3_sub2 (cam, tmp);

	/* Some debugging info */
#if defined (VERBOSE)
	vec_set_fmt ("%12.4g ");
	printf ("cam: ");
	vec3_print_eol (stdout, cam);
	printf ("tgt: ");
	vec3_print_eol (stdout, tgt);
	printf ("ic:  %g %g\n", ic[0], ic[1]);
#endif
	sprintf (out_fn, "%s%04d.txt", options->output_prefix, a);
	drr_write_projection_matrix (vol, cam, tgt, vup, 
				     sid, ic, ps, ires, out_fn);
	printf ("output format %s\n", &options->output_format);
	if (!strcmp(&options->output_format,"pgm")) {
	    sprintf (out_fn, "%s%04d.pgm", options->output_prefix, a);
	} else if (!strcmp(&options->output_format,"pfm")){
	    sprintf (out_fn, "%s%04d.pfm", options->output_prefix, a);
	} else if (!strcmp(&options->output_format,"raw")){
	    sprintf(out_fn, "%s%04d.raw", options->output_prefix, a);
	} else{
	    printf("Error: Undefined output format");
	}
	sprintf (multispectral_fn, "%s%04d.msd", options->output_prefix, a);
	drr_render_volume_perspective (vol, cam, tgt, vup, 
				       sid, ic, ps, 
				       ires, 
				       out_fn, multispectral_fn, 
				       options);
    }
}

void
set_isocenter (Volume* vol, MGHDRR_Options* options)
{
    vol->xmin += options->isocenter[0];
    vol->xmax += options->isocenter[0];
    vol->ymin += options->isocenter[1];
    vol->ymax += options->isocenter[1];
    vol->zmin += options->isocenter[2];
    vol->zmax += options->isocenter[2];
}

int main(int argc, char* argv[])
{
    Volume* vol;
    MGHDRR_Options options;

    parse_args (&options, argc, argv);

    vol = read_mha (options.input_file);
    volume_convert_to_float (vol);
    set_isocenter (vol, &options);

#if defined (PREPROCESS_ATTENUATION)
    preprocess_attenuation (vol);
#endif
    drr_render_volumes (vol, &options);

    return 0;
}
