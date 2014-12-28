/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information

   Algorithm c (modified Hong algorithm)
   -------------------------------------
   Stage 1(a): Compute pencil beam in standard grid
   (z direction is pdd in water)
   (x-y direction is scatter in water, or just 1-d with distance)

   Stage 1(b): Compute RPL in interpolated coordinate system
   (z axis is beam axis)
   (x-y perpendicular to beam, arbitrary v-up vector)

   Stage 2: For each voxel
   a) Look up primary in RPL grid
   b) Convolve to find scatter within x-y axis of primary grid (ignoring tilt)
   ----------------------------------------------------------------------- */
#include "plmdose_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aperture.h"
#include "dose_volume_functions.h"
#include "interpolate.h"
#include "plm_image.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "proj_matrix.h"
#include "proj_volume.h"
#include "ray_data.h"
#include "ray_trace.h"
#include "rpl_volume.h"
#include "rt_beam.h"
#include "rt_depth_dose.h"
#include "rt_dose.h"
#include "rt_lut.h"
#include "rt_parms.h"
#include "rt_plan.h"
#include "rt_sobp.h"
#include "threading.h"
#include "volume.h"

#define VERBOSE 1
#define PROGRESS 1
//#define DEBUG_VOXEL 1
//#define DOSE_GAUSS 1

#if defined (commentout)
static bool voxel_debug = false;
#endif

/* This function is used to rotate a point about a ray in an orbit
 * perpendicular to the ray.  It is assumed that the arbitrary axis of
 * rotation (ray) originates at the Cartesian origin.
 */
static void
rotate_about_ray (
    double *xyz_new,    /* rotated point */
    double *xyz,        /* point to rotate */
    double t,           /* angle of rotation */
    double *ray         /* axis of rotation */
)
{
    double u[3];
    double v[3];
    double w[3];
    double tmp[3] = {0.0, 0.0, 1.0};

    double M[12];

#if defined (commentout)
    double M[16];
#endif

    /* Generate coordinate system */
    vec3_copy (w, ray);
    vec3_normalize1 (w);
    vec3_cross (v, w, tmp);
    vec3_normalize1 (v);
    vec3_cross (u, v, w);

    /* Build the composite matrix
     *   -- Axis rotation: W coincident Z
     *   -- Rotates about Z by theta radians
     *   -- Undoes axis rotation (Z -> W)
     */
    M[4*0 + 0] = u[0]*u[0]*cos(t) + u[0]*v[0]*sin(t) - u[0]*v[0]*sin(t) + v[0]*v[0]*cos(t) + w[0]*w[0];
    M[4*0 + 1] = u[0]*u[1]*cos(t) + u[0]*v[1]*sin(t) - u[1]*v[0]*sin(t) + v[0]*v[1]*cos(t) + w[0]*w[1];
    M[4*0 + 2] = u[0]*u[2]*cos(t) + u[0]*v[2]*sin(t) - u[2]*v[0]*sin(t) + v[0]*v[2]*cos(t) + w[0]*w[2];
    M[4*0 + 3] = 0;

    M[4*1 + 0] = u[1]*u[0]*cos(t) + u[1]*v[0]*sin(t) - u[0]*v[1]*sin(t) + v[1]*v[0]*cos(t) + w[1]*w[0];
    M[4*1 + 1] = u[1]*u[1]*cos(t) + u[1]*v[1]*sin(t) - u[1]*v[1]*sin(t) + v[1]*v[1]*cos(t) + w[1]*w[1]; 
    M[4*1 + 2] = u[1]*u[2]*cos(t) + u[1]*v[2]*sin(t) - u[2]*v[1]*sin(t) + v[1]*v[2]*cos(t) + w[1]*w[2]; 
    M[4*1 + 3] = 0;

    M[4*2 + 0] = u[2]*u[0]*cos(t) + u[2]*v[0]*sin(t) - u[0]*v[2]*sin(t) + v[2]*v[0]*cos(t) + w[2]*w[0];
    M[4*2 + 1] = u[2]*u[1]*cos(t) + u[2]*v[1]*sin(t) - u[1]*v[2]*sin(t) + v[2]*v[1]*cos(t) + w[2]*w[1];
    M[4*2 + 2] = u[2]*u[2]*cos(t) + u[2]*v[2]*sin(t) - u[2]*v[2]*sin(t) + v[2]*v[2]*cos(t) + w[2]*w[2];
    M[4*2 + 3] = 0;

#if defined (commentout)
    M[4*3 + 0] = 0;
    M[4*3 + 1] = 0;
    M[4*3 + 2] = 0;
    M[4*3 + 3] = 1;
#endif

    /* Apply rotation transform */
    mat43_mult_vec3(xyz_new, M, xyz);
}

#if defined (DEBUG_VOXEL)
static void 
debug_voxel (
    double r,             /* current radius */
    double t,             /* current angle */
    double rgdepth,       /* radiographic depth */
    double d,             /* dose from scatter_xyz */
    double* scatter_xyz,  /* space coordinates of scatterer */
    double* ct_xyz,       /* voxel receiving dose */
    double w,             /* gaussian weight (x) */
    double d0,            /* unweighted d */
    double sigma,         /* Gauss kernel stdev */
    double dose           /* aggr dose @ ct_xyz */
)
{
    FILE* fp;

    fp = fopen ("dump_voxel.txt", "a");

    fprintf (fp, "r,t: [%3.3f %1.3f] rgdepth: %2.3f d: %4.1f dist: %3.3f w: %1.4f d0: %4.3f sigma: %1.4f dose: %4.4f\n",
             r, t, rgdepth, d, vec3_dist(scatter_xyz, ct_xyz), w, d0, sigma, dose);

    fclose (fp);
}
#endif

/* This computes the Highland scattering radius due to Coulombic interactions.
 * 
 * This is valid only for an "infinitely thick" medium such as the patient.  A
 * different approximation is used to find the radial scattering in thin
 * degraders.
 */
static double
highland (
    double rgdepth,
    Rt_beam* beam
)
{
#if defined (commentout)
    float rad_length = 1.0;     /* Radiation length of material (g/cm2) */
    float density    = 1.0;     /* Density of material (g/cm3)          */
    float p          = 0.0;     /* Ion momentum (passed in)          */
    float v          = 0.0;     /* Ion velocity (passed in)          */
    float sum        = 0.0;
    float i, tmp;

    for (i=0; i<rgdepth; i+=rgdepth/1000.0f) {
        tmp = ((rgdepth - i) / (p*v));
        sum += (tmp * tmp) * (density / rad_length) * (rgdepth/1000.0f);

        /* Update p and v here */
    }

    sum = sqrtf(sum);
//    printf ("%f\n", 14.1 * (1 + (1/9) * log10(rgdepth/rad_length)) * sum);

    return 14.1 * (1 + (1/9) * log10(rgdepth/rad_length)) * sum; 
#endif

#if defined (commentout) /* MOVE TO BEAM or SOBP class */

    /* This is just a normalization I used to use instead
     * of the Highland approximation */
    return 3.0 * (rgdepth - beam->sobp->d_lut[0]) 
        / (beam->dmax - beam->sobp->d_lut[0]);
#endif
    return 0;
}

static double
highland_maxime_aperture_theta0 (
    double rgdepth,
    Rt_beam* beam
)
{
    float energy = 158.6;		/*Beam energy (MeV)*/
    float mc2 = 939.4; /* proton mass at rest (MeV) */
    float c = 299792458; /* speed of light (m/s2) */
    float rad_length = 36.08;     /* Radiation length of material (g/cm2) */
    float density    = 1.0;     /* Density of material (g/cm3) */
    float p = 0.0;     /* Proton momentum (passed in)          */
    float v = 0.0;     /* Proton velocity (passed in)          */
    float range = 0;			/* Mean range of the proton beam (g/cm2) */
    float stop = 0;				/* stopping power energy (MeV.cm2/g) */
	
    float sum = 0.0;			/* integration expression */
    float step = 0.1;			/*step of the integration along the pathway (cm)*/

    float function_to_be_integrated = 0.0; /* expression to be integrated on dz, second part of the highland's formula */

    UNUSED_VARIABLE (density);
    UNUSED_VARIABLE (range);

    range = getrange(energy);

    /* in the Hong algorithm, rgdepth is in cm but given in mm by plastimatch
       integration of the integrale part of the highland's formula */
    rgdepth = rgdepth/10;   

    for (float i = 0; i <= rgdepth && energy > 0.5; i+=step)
    {
		/* p & v are updated */

        p= sqrt(2*energy*mc2+energy*energy)/c; // in MeV.s.m-1
        v= c*sqrt(1-pow((mc2/(energy+mc2)),2)); //in m.s-1
		/*integration*/

        function_to_be_integrated = 1/(pow(p*v,2)*rad_length);
        sum += function_to_be_integrated*step;

		/* energy is updated after passing through dz */
        stop = getstop(energy);
        energy = energy - stop*step;
    }

    return 14.1 * (1 + (1/9) * log10(rgdepth/rad_length)) * sqrt(sum); 
}

static double
highland_maxime_patient_theta_pt (
    double rgdepth,
    Rt_beam* beam
)
{
    float energy = 85;		/*Beam energy (MeV)*/
    float mc2 = 939.4;          /* proton mass at rest (MeV) */
    float c = 299792458;        /* speed of light (m/s2) */
    float rad_length = 36.08;   /* Radiation length of material (g/cm2) */
    float density    = 1.0;     /* Density of material (g/cm3) !!!!!!!!!! to be determined!! */
    float p = 0.0;              /* Proton momentum (passed in)          */
    float v = 0.0;              /* Proton velocity (passed in)          */
    float range = 0;		/* Mean range of the proton beam (g/cm2) */
    float stop = 0;		/* stopping power energy (MeV.cm2/g) */
	
    float sum = 0.0;		/* integration expression */

    float step = 0.1;		/*step of the integration along the pathway (cm)*/

    float function_to_be_integrated = 0.0; /* expression to be integrated on dz, second part of the highland's formula */

    UNUSED_VARIABLE (range);

    rgdepth = rgdepth /10; /* rgdepth is given in mm by plastimatch, but is in cm in the hong algorithm */

    range = getrange(energy);

    /* integration of the integrale part of the highland's formula */

    for (float i = 0; i <= rgdepth && energy > 0.5; i+=step)
    {
	/* p & v are updated */

        p= sqrt(2*energy*mc2+energy*energy)/c; // in MeV.s.m-1
        v= c*sqrt(1-pow((mc2/(energy+mc2)),2)); //in m.s-1
	/*integration*/

        function_to_be_integrated = (pow(((rgdepth-i)/(p*v)),2)* density / rad_length);
        sum += function_to_be_integrated*step;

	/* energy is updated after passing through dz */
        stop = getstop(energy);
        energy = energy - stop*step;
    }

    //printf(" theta: %lg",  14.1 * (1 + (1/9) * log10(rgdepth/rad_length)) * sqrt(sum) * 10 * rgdepth);
    return 14.1 * (1 + (1/9) * log10(rgdepth/rad_length)) * sqrt(sum) * 10; // y0 * 10 (cm->mm)
}

static double
gaus_kernel (
    double* p,
    double* ct_xyz,
    double sigma
)
{
    double w1, w2;
    double denom;
    double sigma2;

    // kludge to prevent evaluations 
    // greater than 1
    if (sigma < 0.4) {
        sigma = 0.4;
    }

    sigma2 = sigma * sigma;

    denom = 2.0f * M_PI * sigma2;
    denom = sqrt (denom);
    denom = 1.0f / denom;

    w1 = denom * exp ( (-1.0*p[0]*p[0]) / (2.0f*sigma2) );
    w2 = denom * exp ( (-1.0*p[1]*p[1]) / (2.0f*sigma2) );

    return w1 * w2;
}

static double
off_axis_maxime (
    double r,
    double sigma_srm /*,
	double sigma_pt*/
)
{
    double denom;
    double sigma_tot2;

    sigma_tot2 = /*sigma_source +*/ sigma_srm * sigma_srm /*+ sigma_pt * sigma_pt*/; /* !! source !! and sigma patient*/

    denom = 1/ (2.0f * M_PI * sigma_tot2);
    return denom * exp (-1.0*r*r / (2.0f*sigma_tot2));  /* Off-axis term */
}


/* This function should probably be marked for deletion once dose_scatter() is
 * working properly.  GCS: This funcion is useful for debugging.  Let's keep
 * it as flavor 'a'.
 */
double
dose_direct (
    double* ct_xyz,             /* voxel to dose */
    Rt_beam* beam
)
{
    /* Find radiological depth at voxel ct_xyz */
    double rgdepth = beam->rpl_vol->get_rgdepth (ct_xyz); 

    /* The voxel was not hit directly by the beam */
    if (rgdepth <= 0.0f) {
        return 0.0f;
    }

#if defined (commentout)
    printf ("RGD [%g %g %g] = %f, %f\n", 
        ct_xyz[0], ct_xyz[1], ct_xyz[2], rgdepth,
        beam->beam->lookup_sobp_dose (rgdepth));
#endif

    /* return the dose at this radiographic depth */
    return (double) beam->lookup_sobp_dose ((float)rgdepth);
}

double
dose_debug (
    double* ct_xyz,             /* voxel to dose */
    Rt_beam* beam
)
{
#if defined (commentout)
    return rpl_volume_get_rgdepth (beam->rpl_vol, ct_xyz);
#endif

    /* Find radiological depth at voxel ct_xyz */
    return beam->rpl_vol->get_rgdepth (ct_xyz);
}

/* Accounts for small angle scattering due to Columbic interactions */
double
dose_scatter (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    Rt_beam* beam
)
{
    const Aperture::Pointer& ap = beam->get_aperture();
    Rpl_volume*   rpl_vol = beam->rpl_vol;

    double rgdepth;
    double sigma;

    double r, t;
    double r_step, t_step;
    double r_max;

    double sp_pos[3] = {0.0, 0.0, 0.0};
    double scatter_xyz[4] = {0.0, 0.0, 0.0, 1.0};
    double proj_xy[2] = {0.0, 0.0};
    double sctoct[3] = {0.0, 0.0, 0.0};
    double tmp[3] = {0.0, 0.0, 0.0};

    double d = 0.0f;
    double dose = 0.0f;
    double w;

    int debug = 0;

    double dmax = beam->get_sobp_maximum_depth ();

#if defined (DEBUG_VOXEL)
    double d0;

    //int watch_ijk[3] = {0, 255, 67};  // entry
    //int watch_ijk[3] = {134, 256, 67};  // bragg peak
    //int watch_ijk[3] = {23, 255, 67};    // "stem"

    int watch_ijk[3] = {20, 19, 19};

    if (ct_ijk[0] == watch_ijk[0] &&
        ct_ijk[1] == watch_ijk[1] &&
        ct_ijk[2] == watch_ijk[2]) {

        printf ("Watching voxel [%i %i %i]\n", watch_ijk[0], watch_ijk[1], watch_ijk[2]);
        debug = 1;
    }
#endif 

    /* Get approximation for scatterer search radius
     * NOTE: This is not used to define the Gaussian
     */
#if defined (commentout)
    rgdepth = rpl_volume_get_rgdepth (rpl_vol, ct_xyz);
#endif
    rgdepth = rpl_vol->get_rgdepth (ct_xyz);

    if (debug) {
//        printf ("rgdepth = %f\n", rgdepth);
    }

    /* If the voxel was not hit *directly* by the beam, there is still a
     * chance that it was hit by scatterers generated by a neighbor who *was*
     * hit directly by the beam.  As a result, we cannot obtain a resonable
     * estimate, so we assume the largest scattering radius.
     */
    if (rgdepth < 0.0) {
        if (beam->get_detail() == 0) {
            rgdepth = dmax;
        }
        else if (beam->get_detail() == 1) {
            /* User wants to ignore "scatter only" dose */
            if (debug) {
//                printf ("Voxel culled by detail flag\n");
            }
            return 0.0f;
        }
        else {
            rgdepth = dmax;
        }
    }

    sigma = highland (rgdepth, beam);
    r_max = 3.0*sigma;

    r_step = 1.00;          // mm
    t_step = M_PI / 8.0f;   // radians

    if (debug) {
        printf ("sigma = %f\n", sigma);
        printf ("r_max = %f\n", r_max);
        printf ("r_step = %f\n", r_step);
        printf ("t_step = %f\n", t_step);
    }

    /* Step radius */
    for (r = 0; r < r_max; r += r_step) {
        vec3_copy (sp_pos, ct_xyz);
        vec3_scale3 (tmp, ap->pdn, r);
        vec3_add2 (sp_pos, tmp);

        /* Step angle */
        for (t = 0.0f; t < 2.0*M_PI; t += t_step) {

            rotate_about_ray (
                scatter_xyz,  // O: new xyz coordinate
                sp_pos,       // I: init xyz coordinate
                t,            // I: angle of rotation
                ct_xyz);      // I: axis of rotation

            /* neighbor (or self) hit by ion beam? */
#if defined (commentout)
            rgdepth = rpl_volume_get_rgdepth (rpl_vol, scatter_xyz);
#endif
            rgdepth = rpl_vol->get_rgdepth (scatter_xyz);

            if (rgdepth < 0.0f) {
                if (debug) {
                    printf ("Voxel culled by rgdepth\n");
                }
                continue;
            } else {
                d = beam->lookup_sobp_dose (rgdepth);
#if defined (DEBUG_VOXEL)
                d0 = d;
#endif
            }

            vec3_sub3 (sctoct, scatter_xyz, ct_xyz);

            proj_xy[0] = vec3_dot (sctoct, ap->prt);
            proj_xy[1] = vec3_dot (sctoct, ap->pdn);

            sigma = highland (rgdepth, beam);

            /* weight by gaussian kernel */
            w = gaus_kernel (proj_xy, ct_xyz, sigma);
            d *= w;

            /* Add to total dose for our target voxel */
            dose += d;

#if defined (DEBUG_VOXEL)
            if (debug) {
                debug_voxel (r, t, rgdepth, d, scatter_xyz, ct_xyz,
                             w, d0, sigma, dose);
            }
#endif

            /* Don't spin at the origin! */
            if (r == 0) {
                break;
            }

        }
    }

    return dose;    
}

double
dose_hong (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    Rt_beam* beam
)
{
    const Aperture::Pointer& ap = beam->get_aperture();
    Rpl_volume* rpl_vol = beam->rpl_vol;

    double rgdepth;
    double sigma;

    double r, t;
    double r_step, t_step;
    double r_max;

    double sp_pos[3] = {0.0, 0.0, 0.0};
    double scatter_xyz[4] = {0.0, 0.0, 0.0, 1.0};
    double proj_xy[2] = {0.0, 0.0};
    double sctoct[3] = {0.0, 0.0, 0.0};
    double tmp[3] = {0.0, 0.0, 0.0};

    double d = 0.0f;
    double dose = 0.0f;
    double w;

    double dmax = beam->get_sobp_maximum_depth ();

    int debug = 0;

#if defined (DEBUG_VOXEL)
    double d0;

    int watch_ijk[3] = {20, 19, 19};

    if (ct_ijk[0] == watch_ijk[0] &&
        ct_ijk[1] == watch_ijk[1] &&
        ct_ijk[2] == watch_ijk[2]) {

        printf ("Watching voxel [%i %i %i]\n", 
            watch_ijk[0], watch_ijk[1], watch_ijk[2]);
        debug = 1;
    }
#endif 

    /* Get approximation for scatterer search radius
     * NOTE: This is not used to define the Gaussian
     */
#if defined (commentout)
    rgdepth = rpl_volume_get_rgdepth (rpl_vol, ct_xyz);
#endif
    rgdepth = rpl_vol->get_rgdepth (ct_xyz);

    if (debug) {
        printf ("rgdepth = %f\n", rgdepth);
    }

    /* If the voxel was not hit *directly* by the beam, there is still a
     * chance that it was hit by scatterers generated by a neighbor who
     * *was* * hit directly by the beam.  As a result, we cannot obtain
     * a resonable estimate, so we assume the largest scattering radius.
     */
    if (rgdepth < 0.0) {
        if (beam->get_detail() == 0) {
            rgdepth = dmax;
        }
        else if (beam->get_detail() == 1) {
            /* User wants to ignore "scatter only" dose */
            if (debug) {
                printf ("Voxel culled by detail flag\n");
            }
            return 0.0f;
        }
        else {
            rgdepth = dmax;
        }
    }

    sigma = highland (rgdepth, beam);
    r_max = 3.0*sigma;

    r_step = 1.00;          // mm
    t_step = M_PI / 8.0f;   // radians

    if (debug) {
        printf ("sigma = %f\n", sigma);
        printf ("r_max = %f\n", r_max);
        printf ("r_step = %f\n", r_step);
        printf ("t_step = %f\n", t_step);
    }

    /* Step radius */
    for (r = 0; r < r_max; r += r_step) {
        vec3_copy (sp_pos, ct_xyz);
        vec3_scale3 (tmp, ap->pdn, r);
        vec3_add2 (sp_pos, tmp);

        /* Step angle */
        for (t = 0.0f; t < 2.0*M_PI; t += t_step) {

            rotate_about_ray (
                scatter_xyz,  // O: new xyz coordinate
                sp_pos,       // I: init xyz coordinate
                t,            // I: angle of rotation
                ct_xyz);      // I: axis of rotation

            /* neighbor (or self) hit by ion beam? */
#if defined (commentout)
            rgdepth = rpl_volume_get_rgdepth (rpl_vol, scatter_xyz);
#endif
            rgdepth = rpl_vol->get_rgdepth (scatter_xyz);

            if (rgdepth < 0.0f) {
                if (debug) {
                    printf ("Voxel culled by rgdepth\n");
                }
                continue;
            } else {
                d = beam->lookup_sobp_dose (rgdepth);
#if defined (DEBUG_VOXEL)
                d0 = d;
#endif
            }

            vec3_sub3 (sctoct, scatter_xyz, ct_xyz);

            proj_xy[0] = vec3_dot (sctoct, ap->prt);
            proj_xy[1] = vec3_dot (sctoct, ap->pdn);

            sigma = highland (rgdepth, beam);

            /* weight by gaussian kernel */
            w = gaus_kernel (proj_xy, ct_xyz, sigma);
            d *= w;

            /* Add to total dose for our target voxel */
            dose += d;

#if defined (DEBUG_VOXEL)
            if (debug) {
                debug_voxel (r, t, rgdepth, d, scatter_xyz, ct_xyz,
                    w, d0, sigma, dose);
            }
#endif

            /* Don't spin at the origin! */
            if (r == 0) {
                break;
            }

        }
    }
    return dose;    

}

double /* to be implemented */
dose_hong_maxime (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    Rt_beam* beam
)
{
    const Aperture::Pointer& ap = beam->get_aperture();
    Rpl_volume* rpl_vol = beam->rpl_vol;

    double rgdepth;
    double sigma;

    double r, t;
    double r_step, t_step;
    double r_max;

    double r_number = 4; // the number of segmentations
    double t_number = 16; 

    double sp_pos[3] = {0.0, 0.0, 0.0};
    double scatter_xyz[4] = {0.0, 0.0, 0.0, 1.0};
    double proj_xy[2] = {0.0, 0.0};
    double sctoct[3] = {0.0, 0.0, 0.0};
    double tmp[3] = {0.0, 0.0, 0.0};

    double center_ct_xyz[3] = {0.0, 0.0, 0.0};

    double axis[3] = {ct_xyz[0]-beam->get_source_position(0),ct_xyz[1]-beam->get_source_position(1),ct_xyz[2]-beam->get_source_position(2)};
    
    double aperture_right[3] = {0.0,1.0,0.0};
    double aperture_down[3] = {0.0,0.0,-1.0};

    double d = 0.0f;
    double dose = 0.0f;
    double w;
    
    /* creation of a vector perpendicular to axis to initiate the rotation around */
    double vector_init[3] = {1.0,0.0,0.0};
    double vector_opposite[3] = {0.0,0.0,0.0};
    double vector_norm_axis[3] = {0.0,0.0,0.0};

    UNUSED_VARIABLE (ap);
    UNUSED_VARIABLE (sp_pos);
    UNUSED_VARIABLE (proj_xy);
    UNUSED_VARIABLE (sctoct);

    rotate_about_ray(vector_opposite,vector_init,M_PI,axis);
    vec3_sub3 (vector_norm_axis,vector_init,vector_opposite);
    vec3_scale2(vector_norm_axis,1/sqrt(vector_norm_axis[0]*vector_norm_axis[0]+vector_norm_axis[1]*vector_norm_axis[1]+vector_norm_axis[2]*vector_norm_axis[2]));
    
     /* Get approximation for scatterer search radius
     * NOTE: This is not used to define the Gaussian
     */
    rgdepth = rpl_vol->get_rgdepth (ct_xyz);
	
	if (rgdepth < 0.0f) {
            dose = 0;
	    rgdepth = rpl_vol->get_rgdepth(center_ct_xyz);
            } else {
                dose = 0;
            }

    /* If the voxel was not hit *directly* by the beam, there is still a
     * chance that it was hit by scatterers generated by a neighbor who
     * *was* * hit directly by the beam.  As a result, we cannot obtain
     * a resonable estimate, so we assume the largest scattering radius.
     */

    sigma = highland_maxime_patient_theta_pt (rgdepth, beam); /*should be highland_patient_theta0 - !! multiplied by 10 to see it */
    r_max = 3.0*sigma;

    r_step = r_max/r_number;

    t_step =2 * M_PI / t_number;   // radians

    /* Step radius */
    for (int i = 0; i < r_number; i++) {
        r = r_step*(i+1);
        vec3_scale3 (tmp, vector_norm_axis, r);

        /* Step angle */
        for (t = 0.0f; t < 2.0*M_PI; t += t_step) {

            rotate_about_ray (
                scatter_xyz,  // O: new xyz coordinate
                tmp,       // I: init xyz coordinate
                t,            // I: angle of rotation
	        axis);      // I: axis of rotation

            /* neighbor (or self) hit by proton beam? */

            vec3_add2(scatter_xyz, ct_xyz);

            rgdepth = rpl_vol->get_rgdepth (scatter_xyz);

            if (rgdepth < 0.0f) {
		d=0;
            } else {
                d = beam->lookup_sobp_dose (rgdepth);

                proj_xy[0] = vec3_dot (scatter_xyz, aperture_right);
                proj_xy[1] = vec3_dot (scatter_xyz, aperture_down);

                sigma = highland_maxime_patient_theta_pt(rgdepth, beam); /* should be the global one: highland_max_patient_theta0 once the density rho problem will be fixed*/

                /* weight by gaussian kernel */
                w = off_axis_maxime (r, sigma);
                d *= M_PI*(pow(r,2)-pow(r-r_step, 2))/t_number*w; //integration of the pencil beams of this area
			
                /* Add to total dose for our target voxel */
                dose += d;
	    }


            /* Don't spin at the origin! */
            if (r == 0) {
                break;
            }
	}
	r += r_step;
    }
    return dose;    
}

double
dose_hong_sharp (
    double* ct_xyz,             /* voxel to dose */
    Rt_beam* beam
)
{
    double value = beam->rpl_dose_vol->get_rgdepth(ct_xyz);
    /* return the dose at this radiographic depth */
    if (value < 0) {return 0;}
    else {return value;}
}

void
compute_dose_ray_desplanques(Volume* dose_volume, Volume::Pointer ct_vol, Rpl_volume* rpl_volume, Rpl_volume* sigma_volume, Rpl_volume* ct_rpl_volume, Rt_beam* beam, Volume::Pointer final_dose_volume, const Rt_depth_dose* ppp, float normalization_dose)
{
    if (ppp->weight <= 0)
    {
        return;
    }
    int ijk_idx[3] = {0,0,0};
    int ijk_travel[3] = {0,0,0};
    double xyz_travel[3] = {0.0,0.0,0.0};

    double spacing[3] = { (double) (dose_volume->spacing[0]), (double) (dose_volume->spacing[1]), (double) (dose_volume->spacing[2])};
    int ap_ij[2] = {1,0};
    int dim[2] = {0,0};

    double ray_bev[3] = {0,0,0};

    double xyz_ray_center[3] = {0.0, 0.0, 0.0};
    double xyz_ray_pixel_center[3] = {0.0, 0.0, 0.0};

    double entrance_bev[3] = {0.0f, 0.0f, 0.0f}; // coordinates of intersection with the volume in the bev frame
    double entrance_length = 0;
    double distance = 0; // distance from the aperture to the POI
    double tmp[3] = {0.0f, 0.0f, 0.0f};

    double PB_density = 1/(rpl_volume->get_aperture()->get_spacing(0) * rpl_volume->get_aperture()->get_spacing(1));

    double dose_norm = get_dose_norm('f', ppp->E0, PB_density); //the Hong algorithm has no PB density, everything depends on the number of sectors

    double ct_density = 0;
    double sigma = 0;
    int sigma_x3 = 0;
    double rg_length = 0;
    double radius = 0;

    float central_axis_dose = 0;
    float off_axis_factor = 0;

    int idx = 0; // index to travel in the dose volume
    int idx_bev = 0; // second index for reconstructing the final image
    bool test = true;
    bool* in = &test;

    double vec_antibug_prt[3] = {0.0,0.0,0.0};

    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    dim[0] = sigma_volume->get_aperture()->get_dim(0);
    dim[1] = sigma_volume->get_aperture()->get_dim(1);

    float* img = (float*) dose_volume->img;
    float* ct_img = (float*) ct_vol->img;
    float* rpl_image = (float*) rpl_volume->get_vol()->img;

    double dist = 0;
    int offset_step = 0;

    for (int i = 0; i < dim[0]*dim[1]; i++)
    {
		Ray_data* ray_data = &rpl_volume->get_Ray_data()[i];

        ap_ij[1] = i / dim[0];
        ap_ij[0] = i- ap_ij[1]*dim[0];

        vec3_cross(vec_antibug_prt, rpl_volume->get_aperture()->pdn, rpl_volume->get_proj_volume()->get_nrm());

        ray_bev[0] = vec3_dot(ray_data->ray, vec_antibug_prt);
        ray_bev[1] = vec3_dot(ray_data->ray, rpl_volume->get_aperture()->pdn);
        ray_bev[2] = -vec3_dot(ray_data->ray, rpl_volume->get_proj_volume()->get_nrm()); // ray_beam_eye_view is already normalized

        /* Calculation of the coordinates of the intersection of the ray with the clipping plane */
        entrance_length = vec3_dist(rpl_volume->get_proj_volume()->get_src(), ray_data->cp);
        entrance_length += (double) ray_data->step_offset * rpl_volume->get_proj_volume()->get_step_length ();

        vec3_copy(entrance_bev, ray_bev);
        vec3_scale2(entrance_bev, entrance_length);

        if (ray_bev[2]  > DRR_BOUNDARY_TOLERANCE)
        {
          for(int k = 0; k < dose_volume->dim[2] ;k++)
            {
                find_xyz_center(xyz_ray_center, ray_bev, dose_volume->offset[2],k, dose_volume->spacing[2]);
                distance = vec3_dist(xyz_ray_center, entrance_bev);

                ct_density = ct_rpl_volume->get_rgdepth(ap_ij, distance);
				
                if (ct_density <= 0) // no medium, no dose... (air = void)
                {
                    continue;
                }
                else
                {
                    rg_length = rpl_volume->get_rgdepth(ap_ij, distance);
                    central_axis_dose = ppp->lookup_energy((float)rg_length) * ct_density;

                    sigma = sigma_volume->get_rgdepth(ap_ij, distance);
                    sigma_x3 = (int) ceil(3 * sigma);
                    rg_length = rpl_volume->get_rgdepth(ap_ij, distance);

                    /* We defined the grid to be updated, the pixels that receive dose from the ray */
                    /* We don't check to know if we are still in the matrix because the matrix was build to contain all pixels with a 3 sigma_max margin */
                    find_ijk_pixel(ijk_idx, xyz_ray_center, dose_volume);
                    
                    i_min = ijk_idx[0] - sigma_x3;
                    i_max = ijk_idx[0] + sigma_x3;
                    j_min = ijk_idx[1] - sigma_x3;
                    j_max = ijk_idx[1] + sigma_x3;

                    central_axis_dose = beam->lookup_sobp_dose((float) rg_length);
                    
                    for (int i2 = i_min; i2 <= i_max; i2++)
                    {
                        for (int j2 = j_min; j2 <= j_max; j2++)
                        {
                            if (i2 < 0 || j2 < 0 || i2 >= dose_volume->dim[0] || j2 >= dose_volume->dim[1])
                            {
                                continue;
                            }
                            idx = i2 + (dose_volume->dim[0] * (j2 + dose_volume->dim[1] * k));
                            ijk_travel[0] = i2;
                            ijk_travel[1] = j2;
                            ijk_travel[2] = k;

                            find_xyz_from_ijk(xyz_travel,dose_volume,ijk_travel);
                            
                            radius = vec3_dist(xyz_travel,xyz_ray_center); 
                            if (sigma == 0)
                            {
                                off_axis_factor = 1;
                            }
                            else if (radius > sqrt(0.25 * spacing[0] * spacing [0] + 0.25 * spacing[1] * spacing[1]) + 3 * sigma )
                            {
                                off_axis_factor = 0;
                            }
                            else
                            {
                                off_axis_factor = double_gaussian_interpolation(xyz_ray_center, xyz_travel,sigma, spacing);
                            }
                            // SOBP is weighted by the weight of the 
                            // pristine peak
                            img[idx] += normalization_dose 
                                * beam->get_beam_weight() 
                                * central_axis_dose 
                                * off_axis_factor 
                                * (float) ppp->weight 
                                / dose_norm; 
                        }
                    }
                }
            }
        }
        else
        {
            printf("Ray[%d] is not directed forward: z,x,y (%lg, %lg, %lg) \n", i, ray_data->ray[0], ray_data->ray[1], ray_data->ray[2]);
        }
    }

    float* final_dose_img = (float*) final_dose_volume->img;

    int ijk[3] = {0,0,0};
    float ijk_bev[3] = {0,0,0};
    int ijk_bev_trunk[3];
    double xyz_room[3] = {0.0,0.0,0.0};
    float xyz_bev[3] = {0.0,0.0,0.0};

    plm_long mijk_f[3];
    plm_long mijk_r[3];

    float li_frac1[3];
    float li_frac2[3];

    plm_long ct_dim[3] = {ct_vol->dim[0], ct_vol->dim[1], ct_vol->dim[2]};
    plm_long dose_bev_dim[3] = { dose_volume->dim[0], dose_volume->dim[1], dose_volume->dim[2]};

    for (ijk[0] = 0; ijk[0] < ct_dim[0]; ijk[0]++)
    {
        for (ijk[1] = 0; ijk[1] < ct_dim[1]; ijk[1]++)
        {
            for (ijk[2] = 0; ijk[2] < ct_dim[2]; ijk[2]++)
            {
                idx = ijk[0] + ct_dim[0] *(ijk[1] + ijk[2] * ct_dim[1]);
                if ( ct_img[idx] > -1000) // in air we have no dose, we let the voxel number at 0!
                {   
                    final_dose_volume->get_xyz_from_ijk(xyz_room, ijk);

                    /* xyz contains the coordinates of the pixel in the room coordinates */
                    /* we now calculate the coordinates of this pixel in the dose_volume coordinates */
                    /* need to be fixed after the extrinsic homogeneous coordinates is fixed */

                    vec3_sub3(tmp, rpl_volume->get_proj_volume()->get_src(), xyz_room);
                   
                    xyz_bev[0] = (float) -vec3_dot(tmp, vec_antibug_prt);
                    xyz_bev[1] = (float) -vec3_dot(tmp, rpl_volume->get_aperture()->pdn);
                    xyz_bev[2] = (float) vec3_dot(tmp, rpl_volume->get_proj_volume()->get_nrm());

                    dose_volume->get_ijk_from_xyz(ijk_bev,xyz_bev, in);
                    if (*in == true)
                    {
                        dose_volume->get_ijk_from_xyz(ijk_bev_trunk, xyz_bev, in);

                        idx_bev = ijk_bev_trunk[0] + ijk_bev[1]*dose_volume->dim[0] + ijk_bev[2] * dose_volume->dim[0] * dose_volume->dim[1];
                        li_clamp_3d(ijk_bev, mijk_f, mijk_r, li_frac1, li_frac2, dose_volume);
                                                
                        final_dose_img[idx] += li_value(li_frac1[0], li_frac2[0], li_frac1[1], li_frac2[1], li_frac1[2], li_frac2[2], idx_bev, img, dose_volume);
                    }
                    else
                    {
                        final_dose_img[idx] += 0;
                    }
                }
            }   
        }     
    }
    return;
}

void 
compute_dose_ray_sharp (
    const Volume::Pointer ct_vol, 
    const Rpl_volume* rpl_volume, 
    const Rpl_volume* sigma_volume, 
    const Rpl_volume* ct_rpl_volume, 
    const Rt_beam* beam, 
    Rpl_volume* rpl_dose_volume, 
    const Aperture::Pointer ap, 
    const Rt_depth_dose* ppp, 
    const int* margins, 
    float normalization_dose
)
{
    int ap_ij_lg[2] = {0,0};
    int ap_ij_sm[2] = {0,0};
    int dim_lg[3] = {0,0,0};
    int dim_sm[3] = {0,0,0};

    double ct_density = 0;
    double sigma = 0;
    double sigma_x3 = 0;
    double rg_length = 0;

    double central_ray_xyz[3] = {0,0,0};
    double travel_ray_xyz[3] = {0,0,0};

    float central_axis_dose = 0;
    float off_axis_factor = 0;

    double PB_density = 1 / (rpl_volume->get_aperture()->get_spacing(0) * rpl_volume->get_aperture()->get_spacing(1));

    double dose_norm = get_dose_norm ('g', ppp->E0, PB_density);
    //the Hong algorithm has no PB density, everything depends on the number of sectors

    int idx2d_sm = 0;
    int idx2d_lg = 0;
    int idx3d_sm = 0;
    int idx3d_lg = 0;
    int idx3d_travel = 0;

    double minimal_lateral = 0;
    double lateral_step[2] = {0,0};
    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;

    dim_lg[0] = rpl_dose_volume->get_vol()->dim[0];
    dim_lg[1] = rpl_dose_volume->get_vol()->dim[1];
    dim_lg[2] = rpl_dose_volume->get_vol()->dim[2];

    dim_sm[0] = rpl_volume->get_vol()->dim[0];
    dim_sm[1] = rpl_volume->get_vol()->dim[1];
    dim_sm[2] = rpl_volume->get_vol()->dim[2];

    float* rpl_img = (float*) rpl_volume->get_vol()->img;
    float* sigma_img = (float*) sigma_volume->get_vol()->img;
    float* rpl_dose_img = (float*) rpl_dose_volume->get_vol()->img;
    float* ct_rpl_img = (float*) ct_rpl_volume->get_vol()->img;

    double dist = 0;
    double radius = 0;

    /* Creation of the rpl_volume containing the coordinates xyz (beam eye view) and the CT density vol*/
    std::vector<double> xyz_init (4,0);
    std::vector< std::vector<double> > xyz_coor_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], xyz_init);
    std::vector<double> CT_density_vol (dim_lg[0]*dim_lg[1]*dim_lg[2], 0);
    calculate_rpl_coordinates_xyz (&xyz_coor_vol, rpl_dose_volume);

    for (int m = 0; m < dim_lg[0] * dim_lg[1] * dim_lg[2]; m++)
    {
        rpl_dose_img[m] = 0;
    }

    /* calculation of the lateral steps in which the dose is searched constant with depth */
    std::vector <double> lateral_minimal_step (dim_lg[2],0);
    std::vector <double> lateral_step_x (dim_lg[2],0);
    std::vector <double> lateral_step_y (dim_lg[2],0);

    minimal_lateral = ap->get_spacing(0);
    if (minimal_lateral < ap->get_spacing(1))
    {
        minimal_lateral = ap->get_spacing(1);
    }
    for (int k = 0; k < dim_sm[2]; k++)
    {
        lateral_minimal_step[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * minimal_lateral / rpl_volume->get_aperture()->get_distance();
        lateral_step_x[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * ap->get_spacing(0) / rpl_volume->get_aperture()->get_distance();
        lateral_step_y[k] = (rpl_volume->get_front_clipping_plane() + (double) k) * ap->get_spacing(1) / rpl_volume->get_aperture()->get_distance();
    }
    /* calculation of the dose in the rpl_volume */
    for (ap_ij_lg[0] = margins[0]; ap_ij_lg[0] < rpl_dose_volume->get_vol()->dim[0]-margins[0]; ap_ij_lg[0]++){
        for (ap_ij_lg[1] = margins[1]; ap_ij_lg[1] < rpl_dose_volume->get_vol()->dim[1]-margins[1]; ap_ij_lg[1]++){
            ap_ij_sm[0] = ap_ij_lg[0] - margins[0];
            ap_ij_sm[1] = ap_ij_lg[1] - margins[1];

            idx2d_lg = ap_ij_lg[1] * dim_lg[0] + ap_ij_lg[0];
            idx2d_sm = ap_ij_sm[1] * dim_sm[0] + ap_ij_sm[0];

            Ray_data* ray_data = &rpl_dose_volume->get_Ray_data()[idx2d_lg];
            for (int k = 0; k < dim_sm[2]; k++)
            {
                idx3d_lg = idx2d_lg + k * dim_lg[0]*dim_lg[1];
                idx3d_sm = idx2d_sm + k * dim_sm[0]*dim_sm[1];

                central_ray_xyz[0] = xyz_coor_vol[idx3d_lg][0];
                central_ray_xyz[1] = xyz_coor_vol[idx3d_lg][1];
                central_ray_xyz[2] = xyz_coor_vol[idx3d_lg][2];

                lateral_step[0] = lateral_step_x[k];
                lateral_step[1] = lateral_step_x[k];
                ct_density = (double) ct_rpl_img[idx3d_sm];
                if (ct_density <= 0) // no medium, no dose... (air = void) or we are not in the aperture but in the margins fr the penubras
                {
                    continue;
                }
                rg_length = rpl_img[idx3d_sm];
                central_axis_dose = ppp->lookup_energy(rg_length) * ct_density;

                if (central_axis_dose <= 0) 
                {
                    continue;
                } // no dose on the axis, no dose scattered

                sigma = (double) sigma_img[idx3d_sm];
                        
                sigma_x3 = sigma * 3;

                /* finding the rpl_volume pixels that are contained in the the 3 sigma range */                    
                i_min = ap_ij_lg[0] - (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (i_min < 0 ) {i_min = 0;}
                i_max = ap_ij_lg[0] + (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (i_max > dim_lg[0]-1 ) {i_max = dim_lg[0]-1;}
                j_min = ap_ij_lg[1] - (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (j_min < 0 ) {j_min = 0;}
                j_max = ap_ij_lg[1] + (int) ceil(sigma_x3 / lateral_minimal_step[k]);
                if (j_max > dim_lg[1]-1 ) {j_max = dim_lg[1]-1;}
                for (int i1 = i_min; i1 <= i_max; i1++) {
                    for (int j1 = j_min; j1 <= j_max; j1++) {

                        idx3d_travel = k * dim_lg[0]*dim_lg[1] + j1 * dim_lg[0] + i1;

                        travel_ray_xyz[0] = xyz_coor_vol[idx3d_travel][0];
                        travel_ray_xyz[1] = xyz_coor_vol[idx3d_travel][1];
                        travel_ray_xyz[2] = xyz_coor_vol[idx3d_travel][2];
								
                        radius = vec3_dist(travel_ray_xyz, central_ray_xyz);                            
                        if (sigma == 0)
                        {
                            off_axis_factor = 1;
                        }
                        else if (radius / sigma >=3)
                        {
                            off_axis_factor = 0;
                        }
                        else
                        {
                            off_axis_factor = double_gaussian_interpolation(central_ray_xyz, travel_ray_xyz, sigma, lateral_step);
                        }
                        // SOBP is weighted by the weight of the pristine peak
                        rpl_dose_img[idx3d_travel] += normalization_dose 
                            * beam->get_beam_weight() 
                            * central_axis_dose 
                            * off_axis_factor 
                            * (float) ppp->weight 
                            / dose_norm; 
                    } //for j1
                } //for i1
            } // for k
        } // ap_ij[1]
    } // ap_ij[0]   
}

void compute_dose_ray_shackleford(Volume::Pointer dose_vol, Rt_plan* plan, const Rt_depth_dose* ppp, std::vector<double>* area, std::vector<double>* xy_grid, int radius_sample, int theta_sample)
{
    int ijk[3] = {0,0,0};
    double xyz[4] = {0,0,0,1};
    double xyz_travel[4] = {0,0,0,1};
    double tmp_xy[4] = {0,0,0,1};
    double tmp_cst = 0;

    double dose_norm = get_dose_norm('h', ppp->E0, 1); //the Hong algorithm has no PB density, everything depends on the number of sectors

    int idx = 0;
	
    int ct_dim[3] = {dose_vol->dim[0], dose_vol->dim[1], dose_vol->dim[2]};
    double vec_ud[4] = {0,0,0,1};
    double vec_rl[4] = {0,0,0,1};

    float* ct_img = (float*) plan->get_patient_volume()->img;
    float* dose_img = (float*) dose_vol->img;

    double sigma_travel = 0;
    double sigma_3 = 0;
    double rg_length = 0;
	double ct_density = 0;
    double central_sector_dose = 0;
    double radius = 0;
    double theta = 0;
    double dr = 0;

    vec3_copy(vec_ud, plan->beam->rpl_vol->get_proj_volume()->get_incr_c());
    vec3_normalize1(vec_ud);

    vec3_copy(vec_rl, plan->beam->rpl_vol->get_proj_volume()->get_incr_r());
    vec3_normalize1(vec_rl);

    for (ijk[0] = 0; ijk[0] < ct_dim[0]; ijk[0]++){
        for (ijk[1] = 0; ijk[1] < ct_dim[1]; ijk[1]++){
            for (ijk[2] = 0; ijk[2] < ct_dim[2]; ijk[2]++){
                idx = ijk[0] + ct_dim[0] * (ijk[1] + ct_dim[1] * ijk[2]);

                /* calculation of the pixel coordinates in the room coordinates */
                xyz[0] = (double) dose_vol->offset[0] + ijk[0] * dose_vol->spacing[0];
                xyz[1] = (double) dose_vol->offset[1] + ijk[1] * dose_vol->spacing[1];
                xyz[2] = (double) dose_vol->offset[2] + ijk[2] * dose_vol->spacing[2]; // xyz[3] always = 1.0

                sigma_3 = 3 * plan->beam->sigma_vol_lg->get_rgdepth(xyz);

                if (sigma_3 <= 0)
                {
                    continue;
                }
                else
                {
                    for (int i = 0; i < radius_sample; i++)
                    {
                        for (int j =0; j < theta_sample; j++)
                        {

                            vec3_copy(xyz_travel, xyz);

                            /* calculation of the center of the sector */
                            vec3_copy(tmp_xy, vec_ud);
                            tmp_cst = (double) (*xy_grid)[2*(i*theta_sample+j)] * sigma_3; // xy_grid is normalized to a circle of radius sigma x 3 = 1
                            vec3_scale2(tmp_xy, tmp_cst);
                            vec3_add2(xyz_travel,tmp_xy);

                            vec3_copy(tmp_xy, vec_rl);
                            tmp_cst = (double) (*xy_grid)[2*(i*theta_sample+j)+1] * sigma_3;
                            vec3_scale2(tmp_xy, tmp_cst);
                            vec3_add2(xyz_travel,tmp_xy);
							
                            rg_length = plan->beam->rpl_vol->get_rgdepth(xyz_travel);
							ct_density = plan->beam->ct_vol_density_lg->get_rgdepth(xyz_travel);
							
                            if (rg_length <= 0)
                            {
                                continue;
                            }
                            else
                            {
                                /* the dose from that sector is summed */
                                sigma_travel = plan->beam->sigma_vol->get_rgdepth(xyz_travel);
                                radius = vec3_dist(xyz, xyz_travel);
								
                                if (sigma_travel < radius / 3 || (plan->beam->get_aperture()->have_aperture_image() == true && plan->beam->aperture_vol->get_rgdepth(xyz_travel) < 0.999)) 
                                {
                                    continue;
                                }
                                else
                                {
                                    central_sector_dose = plan->beam->lookup_sobp_dose((float) rg_length)* ct_density * (1/(sigma_travel*sqrt(2*M_PI)));
                                    dr = sigma_3 / (2* radius_sample);
                                    // * is normalized to a radius =1, 
                                    // need to be adapted to a 3_sigma 
                                    // radius circle
                                    dose_img[idx] += 
                                        plan->get_normalization_dose() 
                                        * plan->beam->get_beam_weight() 
                                        * central_sector_dose 
                                        * get_off_axis(radius, dr, sigma_3/3) 
                                        * ppp->weight / dose_norm; 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

double get_dose_norm(char flavor, double energy, double PB_density)
{
    if (flavor == 'a')
    {
        return 1; // to be defined
    }
    else if (flavor == 'f')
    {
        return PB_density * (30.5363 + 0.21570 * energy - 0.003356 * energy * energy + 0.00000917 * energy * energy * energy);
    }
    else if (flavor == 'g')
    {
      if (energy >= 70)
      {
        return 60.87 -0.2212*energy + 0.0001536 * energy * energy;
      }
      else
      {
        return 156.735 -4.4787 * energy + .060607 * energy * energy -0.000275 * energy * energy * energy;
      }
    }
    else if (flavor == 'h')
    {
      if (energy >= 100)
      {
        return 88.84 -0.3574*energy + .00001284 * energy * energy + 0.000001468 * energy * energy * energy;
      }
      else
      {
        return 303.34 -7.7026 * energy + .09067 * energy * energy - 0.0003862 * energy * energy * energy;
      }
    }
    else
    {
        return 1;
    }
}

