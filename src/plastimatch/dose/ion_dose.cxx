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
#include "ion_sobp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "proj_matrix.h"
#include "ion_beam.h"
#include "ion_dose.h"
#include "ion_parms.h"
#include "ion_plan.h"
#include "rpl_volume.h"
#include "threading.h"
#include "volume.h"

#include "lookup_range.h"
#include "lookup_stop.h"

#define VERBOSE 1
#define PROGRESS 1
//#define DEBUG_VOXEL 1
//#define DOSE_GAUSS 1

#if defined (commentout)
static bool voxel_debug = false;
#endif

/* Call of the look-up table functions used by the dose calculation*/
static double getrange(double energy);
static double getstop(double energy);

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
    Ion_beam* beam
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
    Ion_beam* beam
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
	range = getrange(energy);

	/* integration of the integrale part of the highland's formula */

	for (float i = 0; i <= rgdepth && energy > 1; i+=step)
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
    Ion_beam* beam
)
{
    float energy = 85;		/*Beam energy (MeV)*/
	float mc2 = 939.4; /* proton mass at rest (MeV) */
	float c = 299792458; /* speed of light (m/s2) */
	float rad_length = 36.08;     /* Radiation length of material (g/cm2) */
    float density    = 1.0;     /* Density of material (g/cm3) !!!!!!!!!! to be determined!! */
    float p = 0.0;     /* Proton momentum (passed in)          */
    float v = 0.0;     /* Proton velocity (passed in)          */
    float range = 0;			/* Mean range of the proton beam (g/cm2) */
	float stop = 0;				/* stopping power energy (MeV.cm2/g) */
	
	float sum = 0.0;			/* integration expression */

	float step = 0.1;			/*step of the integration along the pathway (cm)*/

	float function_to_be_integrated = 0.0; /* expression to be integrated on dz, second part of the highland's formula */

	range = getrange(energy);

	/* integration of the integrale part of the highland's formula */

	for (float i = 0; i <= rgdepth && energy > 1; i+=step)
    {
		/* p & v are updated */

        p= sqrt(2*energy*mc2+energy*energy)/c; // in MeV.s.m-1
        v= c*sqrt(1-pow((mc2/(energy+mc2)),2)); //in m.s-1
		/*integration*/

        function_to_be_integrated = (pow(((rgdepth-i)/(p*v)),2)* density / rad_length); // x rho????
        sum += function_to_be_integrated*step;

		/* energy is updated after passing through dz */
        stop = getstop(energy);
        energy = energy - stop*step;
    }

    return 14.1 * (1 + (1/9) * log10(rgdepth/rad_length)) * sqrt(sum) * rgdepth; /* yo * rpl */
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
    double* p,
    double sigma_srm /*,
	double sigma_pt*/
)
{
    double w1, w2;
    double denom;
    double sigma_tot2;

    sigma_tot2 = /*sigma_source +*/ sigma_srm * sigma_srm /*+ sigma_pt * sigma_pt*/; /* !! source !! and sigma patient*/

    denom = 1/ (2.0f * M_PI * sigma_tot2);
    denom = sqrt (denom);
    denom = 1.0f / denom;
    w1 = denom * exp ( (-1.0*p[0]*p[0]) / (2.0f*sigma_tot2) );
    w2 = denom * exp ( (-1.0*p[1]*p[1]) / (2.0f*sigma_tot2) );

    return w1 * w2;  /* Off-axis term */
}


/* This function should probably be marked for deletion once dose_scatter() is
 * working properly.  GCS: This funcion is useful for debugging.  Let's keep
 * it as flavor 'a'.
 */
double
dose_direct (
    double* ct_xyz,             /* voxel to dose */
    const Ion_plan* scene
)
{
    /* Find radiological depth at voxel ct_xyz */
    double rgdepth = scene->rpl_vol->get_rgdepth (ct_xyz); 

    /* The voxel was not hit directly by the beam */
    if (rgdepth <= 0.0f) {
        return 0.0f;
    }

#if defined (commentout)
    printf ("RGD [%g %g %g] = %f, %f\n", 
        ct_xyz[0], ct_xyz[1], ct_xyz[2], rgdepth,
        scene->beam->lookup_sobp_dose (rgdepth));
#endif

    /* return the dose at this radiographic depth */
    return (double) scene->beam->lookup_sobp_dose ((float)rgdepth);
}

double
dose_debug (
    double* ct_xyz,             /* voxel to dose */
    const Ion_plan* scene
)
{
#if defined (commentout)
    return rpl_volume_get_rgdepth (scene->rpl_vol, ct_xyz);
#endif

    /* Find radiological depth at voxel ct_xyz */
    return scene->rpl_vol->get_rgdepth (ct_xyz);
}

/* Accounts for small angle scattering due to Columbic interactions */
double
dose_scatter (
    double* ct_xyz,
    plm_long* ct_ijk,            // DEBUG
    const Ion_plan* scene
)
{
    const Aperture::Pointer& ap = scene->get_aperture();
    Ion_beam*  beam    = scene->beam;
    Rpl_volume*   rpl_vol = scene->rpl_vol;

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
    const Ion_plan* scene
)
{
    const Aperture::Pointer& ap = scene->get_aperture();
    Ion_beam* beam = scene->beam;
    Rpl_volume* rpl_vol = scene->rpl_vol;

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
    int* ct_ijk,            // DEBUG
    const Ion_plan* scene
)
{
	const Aperture::Pointer& ap = scene->get_aperture();
    Ion_beam* beam = scene->beam;
    Rpl_volume* rpl_vol = scene->rpl_vol;

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

	double center_ct_xyz[3] = {0.0,0.0,0.0};
	double axis[3] = {1.0,0.0,0.0};
	double aperture_right[3] = {0.0,1.0,0.0};
	double aperture_down[3] = {0.0,0.0,-1.0};

    double d = 0.0f;
    double dose = 0.0f;
    double w;

    /* Get approximation for scatterer search radius
     * NOTE: This is not used to define the Gaussian
     */
    rgdepth = rpl_vol->get_rgdepth (ct_xyz);
	//printf("\n center rgdepth = %lg - pixel : %lg %lg %lg", rgdepth, ct_xyz[0], ct_xyz[1], ct_xyz[2]);

	if (rgdepth < 0.0f) {
			dose = 0;
			rgdepth = rpl_vol->get_rgdepth(center_ct_xyz);
            } else {
                dose = beam->lookup_sobp_dose (rgdepth);
				//printf("\n dose ok point");
            }
				//printf("\n Direct dose : %lg", dose);

    /* If the voxel was not hit *directly* by the beam, there is still a
     * chance that it was hit by scatterers generated by a neighbor who
     * *was* * hit directly by the beam.  As a result, we cannot obtain
     * a resonable estimate, so we assume the largest scattering radius.
     */

    sigma = 3* highland_maxime_aperture_theta0 (rgdepth, beam); /*should be highland_patient_theta0 - !! multiplied by 10 to see it */
    r_max = 3.0*sigma;

    r_step = r_max/r_number;

    t_step =2 * M_PI / t_number;   // radians

   /*if (debug) {
        printf ("sigma = %f\n", sigma);
        printf ("r_max = %f\n", r_max);
        printf ("r_step = %f\n", r_step);
        printf ("t_step = %f\n", t_step);
    } */

    /* Step radius */
    for (int i = 0; i < r_number; i++) {
        r = r_step*(i+1);
		//printf("\n base1 r= %lg",r);
		vec3_copy (sp_pos, ct_xyz);
		//printf("\n base1 sp_pos: %lg %lg %lg",sp_pos[0],sp_pos[1],sp_pos[2]);
        vec3_scale3 (tmp, aperture_down, r);
		//printf("\n base ap->pdn: %lg %lg %lg - ap->prt: %lg %lg %lg",ap->pdn[0],ap->pdn[1],ap->pdn[2],ap->prt[0],ap->prt[1],ap->prt[2]);
		//printf("\n base2 R = %lg",r);
		//printf("\n base tmp: %lg %lg %lg",tmp[0],tmp[1],tmp[2]);

        vec3_add2 (sp_pos, tmp);
		//printf("\n base2 sp_pos: %lg %lg %lg",sp_pos[0],sp_pos[1],sp_pos[2]);

        /* Step angle */
        for (t = 0.0f; t < 2.0*M_PI; t += t_step) {

            rotate_about_ray (
                scatter_xyz,  // O: new xyz coordinate
                sp_pos,       // I: init xyz coordinate
                t,            // I: angle of rotation
				axis);      // I: axis of rotation

            /* neighbor (or self) hit by proton beam? */

            rgdepth = rpl_vol->get_rgdepth (scatter_xyz);
			//printf("\n init_pos : %lg %lg %lg - t= %lg - axis rot %lg %lg %lg", sp_pos[0],sp_pos[1],sp_pos[2], t, axis[0],axis[1],axis[2]);
			//printf("\n turning rgdepth = %lg - pixel : %lg %lg %lg", rgdepth, scatter_xyz[0], scatter_xyz[1], scatter_xyz[2]);

            if (rgdepth < 0.0f) {
				d=0;
				//printf("X");
            } else {
                d = beam->lookup_sobp_dose (rgdepth);
				//printf("O");

			//printf("\n Dose_at_scattering point : %lg", d);

            vec3_sub3 (sctoct, scatter_xyz, ct_xyz);

            proj_xy[0] = vec3_dot (sctoct, aperture_right);
            proj_xy[1] = vec3_dot (sctoct, aperture_down);

            sigma = 3 * highland_maxime_aperture_theta0(rgdepth, beam); /* should be the global one: highland_max_patient_theta0 once the density rho problem will be fixed*/

            /* weight by gaussian kernel */
            w = off_axis_maxime (proj_xy, sigma /*, sigma patient*/);
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


static double getrange(double energy)
{
    double energy1 = 0;
    double energy2 = 0;
    double range1 = 0;
    double range2 = 0;
	int i=0;

	if (energy >0)
	{
		while (energy >= energy1)
		{
			energy1 = lookup_range[i][0];
			range1 = lookup_range[i][1];

			if (energy >= energy1)
			{
				energy2 = energy1;
				range2 = range1;
			}
			i++;
		}
		return (range2+(energy-energy2)*(range1-range2)/(energy1-energy2));
	}
	else
	{
		return 0;
	}
}

static double getstop(double energy)
{
    double energy1 = 0;
    double energy2 = 0;
    double stop1 = 0;
    double stop2 = 0;
	int i=0;

	if (energy >0)
	{
		while (energy >= energy1)
		{
			energy1 = lookup_stop[i][0];
			stop1 = lookup_stop[i][1];

			if (energy >= energy1)
			{
				energy2 = energy1;
				stop2 = stop1;
			}
			i++;
		}
		return (stop2+(energy-energy2)*(stop1-stop2)/(energy1-energy2));
	}
	else
	{
		return 0;
	}
}