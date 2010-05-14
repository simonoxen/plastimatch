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
//#define PROGRESS 1
#define DOSE_DIRECT
//#define DOSE_GAUSS

typedef struct callback_data Callback_data;
struct callback_data {
    Volume* depth_vol;  /* Radiographic depth volume */
    int* ires;          /* Aperture Dimensions */
    int ap_idx;         /* Current Aperture Coord */
    double accum;       /* Accumulated intensity */
};

typedef struct proton_energy_profile Proton_Energy_Profile;
struct proton_energy_profile {
    float* depth;       /* depth array (mm) */
    float* energy;      /* energy array */
    float dmax;         /* maximum depth */
    float emax;         /* max energy */
    int num_samp;       /* size of arrays */
};


static double
gaus_kernel (
    double x,
    double* ct_xyz,
    double sigma
)
{
    double weight;
    double pi;
    double denom;
    double sigma2;

    pi = 3.14159265f;
    sigma2 = sigma * sigma;

    denom = 2.0f * pi * sigma2;
    denom = sqrt (denom);
    denom = 1.0f / denom;

    weight = denom * exp ( (-1*x*x) / (2.0f*sigma2) );

    return weight;
}


static double
rgdepth_lookup (
        Volume* depth_vol,
        int ap_x,
        int ap_y,
        double dist,
        float ray_step
)
{
    int idx1, idx2;
    int ijk[3];
    double rg1, rg2, rgdepth;
    float* d_img = (float*) depth_vol->img;

    ijk[0] = ap_x;
    ijk[1] = ap_y;
    ijk[2] = dist / ray_step;

    /* Depth behind point */
    idx1 = INDEX_OF (ijk, depth_vol->dim);
    rg1 = d_img[idx1];

    /* Depth in front of point */
    ijk[2]++;
    idx2 = INDEX_OF (ijk, depth_vol->dim);
    rg2 = d_img[idx2];

    dist = dist - floorf (dist);
    
    rgdepth = rg1 + dist * (rg2 - rg1);

    return rgdepth;

#if defined (commentout)
    return d_img[INDEX_OF (ijk, depth_vol->dim)];
#endif

}

static float
energy_lookup (
    float depth,
    Proton_Energy_Profile* pep
)
{
    int i;
    float energy = 0.0f;

    /* Sanity check */
    if (depth < 0 || depth > pep->dmax) {
        return 0.0f;
    }

    /* Find index into profile arrays */
    for (i = 0; i < pep->num_samp; i++) {
        if (pep->depth[i] > depth) {
            i--;
            break;
        } else if (pep->depth[i] == depth) {
            return pep->energy[i];
        }
    }

    /* Use index to lookup and interpolate energy */
    if (i >= 0 || i < pep->num_samp) {
        // linear interpolation
        energy = pep->energy[i]
                + (depth - pep->depth[i])
                * ((pep->energy[i+1] - pep->energy[i]) / (pep->depth[i+1] - pep->depth[i]));
    } else {
        // this should never happen, failsafe
        energy = 0.0f;
    }

    return energy;   
}


static double
get_rgdepth (
    double* ct_xyz,
    double* depth_offset,
    Volume* depth_vol,
    Proton_Energy_Profile* pep,
    Proj_matrix *pmat,
    int* ires,
    double* ap_ul,
    double* incr_r,
    double* incr_c,
    Proton_dose_options *options
)
{
    int ap_x, ap_y, ap_idx;
    double ap_xy[3], ap_xyz[3], tmp[3];
    double dist, depth_rg;

    /* Back project the voxel to the aperture plane */
    mat43_mult_vec3 (ap_xy, pmat->matrix, ct_xyz);

    ap_x = ROUND_INT (pmat->ic[0] + ap_xy[0] / ap_xy[2]);
    ap_y = ROUND_INT (pmat->ic[1] + ap_xy[1] / ap_xy[2]);

    /* Only handle voxels that were hit by the beam */
    if (ap_x < 0 || ap_x >= ires[0] ||
        ap_y < 0 || ap_y >= ires[1]) {
        return -1;
    }

    ap_idx = ap_y * ires[0] + ap_x;

    /* Convert aperture indices into space coords */
    vec3_copy (ap_xyz, ap_ul);
    vec3_scale3 (tmp, incr_r, ap_x);
    vec3_add2 (ap_xyz, tmp);
    vec3_scale3 (tmp, incr_c, ap_y);
    vec3_add2 (ap_xyz, tmp);

    /* Compute distance from aperture to voxel */
    dist = vec3_dist (ap_xyz, ct_xyz);
    dist -= depth_offset[ap_idx];

    /* Retrieve the radiographic depth */
    depth_rg = rgdepth_lookup (depth_vol,
                               ap_x, ap_y,
                               dist,
                               options->ray_step);

    return depth_rg;
}

/* This function should probably be marked for
 * deletion once dose_scatter() is working properly.
 */
static double
dose_direct (
    double* ct_xyz,             /* voxel to dose */
    double* depth_offset,       /* ap to ct_vol dist */
    Volume* depth_vol,          /* radiographic depths */
    Proton_Energy_Profile* pep, /* proton energy profile */
    Proj_matrix *pmat,          /* projection matrix */
    int* ires,                  /* ray cast resolution */
    double* ap_ul,              /* aperture upper-left */
    double* incr_r,             /* ray row to row vector */
    double* incr_c,             /* ray col to col vector */
    Proton_dose_options *options
)
{
    double depth_rg;
    double dose;

    depth_rg = get_rgdepth (ct_xyz,         /* voxel to dose */
                            depth_offset,   /* ap to ct_vol dist */
                            depth_vol,      /* radiographic depths */
                            pep,            /* proton energy profile */
                            pmat,           /* projection matrix */
                            ires,           /* ray cast resolution */
                            ap_ul,          /* aperture upper-left 3D coord */
                            incr_r,         /* distance between ray rows @ aperture */
                            incr_c,         /* distance between ray cols @ aperture */
                            options);       /* contains uniform step size along rays */

    /* The voxel was not hit directly by the beam */
    if (depth_rg < 0.0f) {
        return 0.0f;
    }

    /* Lookup the dose at this radiographic depth */
    dose = energy_lookup (depth_rg, pep);
    
    return dose;

}

static double
dose_scatter (
    double* ct_xyz,
    double* depth_offset,
    Volume* depth_vol,
    Proton_Energy_Profile* pep,
    Proj_matrix *pmat,
    int* ires,
    double* ap_ul,
    double* incr_r,
    double* incr_c,
    double* prt,
    double* pdn,
    Proton_dose_options *options
)
{

    /* Accounts for 1st order small angle scattering
     *
     *   We do this by defining a "scatter plane" that
     *   is centered on the voxel we want to find the
     *   total dose for (direct + scattered dose).
     *
     *   The scatter plane's coordinates are transformed
     *   such that it is parallel to the aperture plane
     *   and normal to the proton beam.
     */


    /* NOTES:
     *   -- ct_xyz[3] contains the 3D real space coordinates
     *      of the voxel being hit *directly* by the proton beam.
     *
     *   -- anything with the sp_ prefix defines the scattering
     *      plane surrounding the ct_xyz voxel.
     *
     *   -- scatter[3] contains the 3D real space coordinates
     *      of the current voxel under consideration within
     *      the scattering plane defined by sp_
     *
     *   -- sctoct[3] is a vector originating at scatter[3]
     *      and ending at ct_xyz[3].  This is needed to
     *      evaluate the Gaussian kernel, which is used to
     *      appropriately weight the contribution of scatter[3]
     *      @ ct_xyz[3].
     */

    double depth_rg;
    double sigma;
    double search_dist;

    /* scatter plane un-transformed */
    int sp_xy[2];    // 2D coordinate in scatter plane
    double sp_step;  // step size in both x & y
    int sp_dim;      // width = height of scatter plane

    /* scatter plain transformed */
    double sp_start[3];   // upper left point in scatter plane
    double sp_incr_x[3];  // x-direction increment vector
    double sp_incr_y[3];  // y-direction increment vector

    double tmp[3];
    double dose = 0.0f;

    int sane;

    /* Get approximation for scatterer search radius
     * NOTE: This is not used to define the Gaussian
     */
    depth_rg = get_rgdepth (ct_xyz, depth_offset,
                            depth_vol, pep,
                            pmat, ires, ap_ul,
                            incr_r, incr_c,
                            options);

    /* If the voxel was not hit *directly* by the beam,
     * there is still a chance that it was hit by
     * scatterers generated by a neighbor who *was*
     * hit directly by the beam.  As a result, we cannot
     * obtain a resonable estimate, so we assume the
     * largest scattering radius.
     */
    if (depth_rg < 0.0) {
//        depth_rg = pep->dmax; // for accuracy
        return 0.0f;            // for speed
    }


    /* For now we are not considering several items in the beam path,
     * so we will approximate the Gaussian bandwidth as linearly
     * related to the radiographic depth.
     */
    sigma = floorf (depth_rg);

    /* We must define a neighborhood around the voxel for which
     * are computing the dose.  All voxels falling within this
     * neighborhood will be checked for dose contributions. This
     * neighborhood is defined as a plane called the "scatter plane."
     * 
     * The plan is to start and end our search 3*sigma from the
     * voxel of interest in both directions.
     */
    search_dist = 3.0*sigma;

    /* We will maintain a uniform sampling along these Gaussian
     * kernel functions in the interest of accuracy.  This will,
     * however, result in slower execution as the beam depth
     * increases.  We can "fix" this later for speed.
     */
    sp_step = 1.00;

    /* calc. steps in transformed scatter plane */
    vec3_scale3 (sp_incr_x, prt, sp_step);
    vec3_scale3 (sp_incr_y, pdn, sp_step);

    /* Scan the scatter plane starting from the top left */
    vec3_copy (sp_start, ct_xyz);
    vec3_scale3 (tmp, prt, - search_dist);  // x
    vec3_add2 (sp_start, tmp);
    vec3_scale3 (tmp, pdn, - search_dist);  // y
    vec3_add2 (sp_start, tmp);

    /* define dimenions of untransformed scatter plane */
    sp_dim = 2*((search_dist + sp_step - 1) / sp_step);

    
#if defined (commentout)
    sane = 0;
#endif
    /* Step along y-dim of scatter plane */
    for (sp_xy[1] = 0; sp_xy[1] < sp_dim; sp_xy[1]++) {
        double sp_pos[3];
        double tmp[3];

        vec3_copy (sp_pos, sp_start);
        vec3_scale3 (tmp, sp_incr_y, (double) sp_xy[1]);
        vec3_add2 (sp_pos, tmp);

        /* Step along x-dim of scatter plane */
        for (sp_xy[0] = 0; sp_xy[0] < sp_dim; sp_xy[0]++) {
            double scatter[3];
            double d;
            double proj_xy[2];
            double sctoct[3];

            vec3_scale3 (tmp, sp_incr_x, (double) sp_xy[0]);
            vec3_add3 (scatter, sp_pos, tmp);

#if defined (commentout)
            /* Sanity check
             *   If we somehow step over the origin,
             *   any dose contributed by the beam directly
             *   interacting with the voxel will be neglected.
             */
            if (vec3_dist (scatter, ct_xyz) < 1.0f) {
                sane = 1;
            }
#endif

            /* neighbor (or self) hit by proton beam? */
            depth_rg = get_rgdepth (scatter,
                                    depth_offset,
                                    depth_vol, pep,
                                    pmat, ires, ap_ul,
                                    incr_r, incr_c,
                                    options);

            if (depth_rg < 0.0f) {
                /*
                 * Neighbor (or self) not hit directly by beam,
                 * so no dose contribution
                 */
                continue;

            } else {
                /* 
                 * Lookup the dose at neighbor (or self's)
                 * radiographic depth
                 */
                d = energy_lookup (depth_rg, pep);
            }

            /* 
             * define a vector from the scatterer to
             * destination voxel at which we are computing
             * the dose.
             *
             * in the event this distance is zero, we are
             * accumulating the dose contribution due to the
             * proton beam itself -- not from a scatterer.
             */
            vec3_sub3 (sctoct, scatter, ct_xyz);

            /*
             * project this vector onto the x and y axis
             * of the transformed scatter plane's
             * coordinate system.
             */
            proj_xy[0] = vec3_dot (sctoct, prt);
            proj_xy[1] = vec3_dot (sctoct, pdn);


            /*
             * the width (or scatterng intensity) of the
             * gaussian kernel function is related to
             * the radiographic depth of the scatterer.
             */
            sigma = depth_rg;
            
            /* apply the gaussian kernel as two 1D operations */
            d *= gaus_kernel (proj_xy[0], ct_xyz, sigma);
            d *= gaus_kernel (proj_xy[1], ct_xyz, sigma);

            /* Add to total dose for our target voxel */
            dose += d;
        }
    }

#if defined (commentout)
    if (!sane) {
        printf ("\nSanity check failed!\nTerminating...\n\n");
        exit(0);
    }
#endif

    return dose;    

}

static void
dump_pep (Proton_Energy_Profile* pep)
{
    FILE* fp;

    int i;

    fp = fopen ("dump_pep.txt", "w");

    for (i = 0; i < pep->num_samp; i++) {
       fprintf (fp, "[%3.2f] %3.2f\n", pep->depth[i], pep->energy[i]);
    }

    fprintf (fp, "    dmax: %3.2f\n", pep->dmax);
    fprintf (fp, "    emax: %3.2f\n", pep->emax);
    fprintf (fp, "num_samp: %i\n", pep->num_samp);

    fclose (fp);
}

static Proton_Energy_Profile* 
load_pep (char* filename)
{
    int i,j;
    char* ptoken;
    char linebuf[128];
    FILE* fp = fopen (filename, "r");
    Proton_Energy_Profile *pep;

    if (!fp) {
        printf ("Error reading proton energy profile.\n");
        printf ("Terminating...\n");
        exit(0);
    }

    // Need to check for a magic number here!!
    // 00001037 ??
    
    // Allocate the pep
    pep = (Proton_Energy_Profile*)malloc(sizeof(Proton_Energy_Profile));

    // Skip the first 4 lines
    for (i=0; i < 4; i++) {
        fgets (linebuf, 128, fp);
    }

    // Line 5 contains the # of samples
    fgets (linebuf, 128, fp);
    sscanf (linebuf, "%i", &pep->num_samp);

    pep->depth = (float*)malloc (pep->num_samp*sizeof(float));
    pep->energy = (float*)malloc (pep->num_samp*sizeof(float));
    
    memset (pep->depth, 0, pep->num_samp*sizeof(float));
    memset (pep->energy, 0, pep->num_samp*sizeof(float));

    // Load in the depths
    // There are 10 samples per line
    for (i = 0, j = 0; i < (pep->num_samp / 10) + 1; i++) {
        fgets (linebuf, 128, fp);

        ptoken = strtok (linebuf, ",\n\0");

        while (ptoken) {
            pep->depth[j++] = (float) strtod (ptoken, NULL);
            ptoken = strtok (NULL, ",\n\0");
        }
    }
    pep->dmax = pep->depth[j-1];
    pep->emax = 0.0f;

    // Load in the energies
    // There are 10 samples per line
    for (i = 0, j = 0; i < (pep->num_samp / 10) + 1; i++) {
        fgets (linebuf, 128, fp);

        ptoken = strtok (linebuf, ",\n\0");

        while (ptoken) {
            pep->energy[j] = (float) strtod (ptoken, NULL);
            if (pep->energy[j] > pep->emax) {
                pep->emax = pep->energy[j];
            }
            ptoken = strtok (NULL, ",\n\0");
            j++;
        }
    }

    return pep;
}

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
    int ct_dims[3], // ct volume dimensions
    int ires[2],    // aperture dimensions
    float ray_step  // uniform ray step size
)
{
    /* TODO: Currently allocates too much
     * memory and is theoretically unsound.
     * Work in mm *then* convert to voxels.
     */

    int dv_dims[3];
    float dv_off[3] = {0.0f, 0.0f, 0.0f};
    float dv_ps[3] = {1.0f, 1.0f, 1.0f};
    float ct_diag;

    ct_diag =  ct_dims[0]*ct_dims[0];
    ct_diag += ct_dims[1]*ct_dims[1];
    ct_diag += ct_dims[2]*ct_dims[2];
    ct_diag = sqrt (ct_diag);

    dv_dims[0] = ires[0];   // rows = aperture rows
    dv_dims[1] = ires[1];   // cols = aperture cols
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
    double* depth_offset,
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

    /* Define unit vector in ray direction */
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

    /* store the distance from aperture to CT_vol for later */
    depth_offset[ap_idx] = vec3_dist (p2, ip1);

    /* init callback data for this ray */
    cd.accum = 0.0f;
    cd.ires = ires;
    cd.depth_vol = depth_vol;
    cd.ap_idx = ap_idx;

    /* get radiographic depth along ray */
    ray_trace_uniform (ct_vol,              // INPUT: CT volume
        vol_limit,                          // INPUT: CT volume bounding box
        &proton_dose_ray_trace_callback,    // INPUT: step action cbFunction
        &cd,                                // INPUT: callback data
        ip1,                                // INPUT: ray starting point
        ip2,                                // INPUT: ray ending point
        options->ray_step);                 // INPUT: uniform ray step size
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
    int ct_ijk[3];
    double ap_xy[3], ap_xyz[3];
    double ct_xyz[3];
    double p1[3];
    double ap_dist = 1000.;
    double nrm[3], pdn[3], prt[3], tmp[3];
    double ic_room[3];
    double ul_room[3];
    double br_room[3];
    double incr_r[3];
    double incr_c[3];
    Volume_limit ct_limit;

    double *depth_offset;
    float* dose_img = (float*) dose_vol->img;


    Proj_matrix *pmat;
    double cam[3] = { options->src[0], options->src[1], options->src[2] };
    double tgt[3] = { options->isocenter[0], options->isocenter[1], 
              options->isocenter[2] };
    double vup[3] = { options->vup[0], options->vup[1], options->vup[2] };
    double ic[2] = { 4.5, 4.5 };
    double ps[2] = { 1., 1. };
    int ires[2] = { 10, 10 };
    Proton_Energy_Profile* pep;

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

    /* Get position of the bottom right pixel on panel */
    vec3_copy (br_room, ul_room);
    vec3_scale3 (tmp, incr_r, (double) (ires[0] - 1));
    vec3_add2 (br_room, tmp);
    vec3_scale3 (tmp, incr_c, (double) (ires[1] - 1));
    vec3_add2 (br_room, tmp);

    /* drr_trace_ray uses p1 & p2, p1 is the camera, p2 is in the 
       direction of the ray inside the aperture plane*/
    vec3_copy (p1, pmat->cam);

#if VERBOSE
    printf ("NRM: %g %g %g\n", nrm[0], nrm[1], nrm[2]);
    printf ("PDN: %g %g %g\n", pdn[0], pdn[1], pdn[2]);
    printf ("PRT: %g %g %g\n", prt[0], prt[1], prt[2]);
    printf ("CAM: %g %g %g\n", pmat->cam[0], pmat->cam[1], pmat->cam[2]);
    printf ("ICR: %g %g %g\n", ic_room[0], ic_room[1], ic_room[2]);
    printf ("INCR_C: %g %g %g\n", incr_c[0], incr_c[1], incr_c[2]);
    printf ("INCR_R: %g %g %g\n", incr_r[0], incr_r[1], incr_r[2]);
    printf ("UL_ROOM: %g %g %g\n", ul_room[0], ul_room[1], ul_room[2]);
    printf ("BR_ROOM: %g %g %g\n", br_room[0], br_room[1], br_room[2]);
#endif

    /* Compute volume boundary box */
    volume_limit_set (&ct_limit, ct_vol);

    /* Create the depth volume */
    depth_vol = create_depth_vol (ct_vol->dim, ires, options->ray_step);

    /* Load proton energy profile specified on commandline */
    pep = load_pep (options->input_pep_fn);

    /* Holds distance from aperture to CT_vol entry point for each ray */
    depth_offset = (double*) malloc (ires[0] * ires[1] * sizeof(double));


    /* Scan through the aperture */
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
        
            proton_dose_ray_trace (depth_vol,    /* O: radiographic depths */
                                   depth_offset, /* O: aperture to vol dists */
                                   ct_vol,       /* I: CT volume */
                                   &ct_limit,    /* I: CT bounding region */
                                   p1,           /* I: @ source */
                                   p2,           /* I: @ aperture */
                                   ires,         /* I: ray cast resolution */
                                   ap_idx,       /* I: linear index of ray in ap */
                                   options);
        }
    }


    if (options->debug) {
        write_mha("depth_vol.mha", depth_vol);
        dump_pep (pep);
    }


    /* Scan through CT Volume */
    for (ct_ijk[2] = 0; ct_ijk[2] < ct_vol->dim[2]; ct_ijk[2]++) {
        for (ct_ijk[1] = 0; ct_ijk[1] < ct_vol->dim[1]; ct_ijk[1]++) {
            for (ct_ijk[0] = 0; ct_ijk[0] < ct_vol->dim[0]; ct_ijk[0]++) {
                double dose;
                int idx;
                
                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->offset[0] + ct_ijk[0] * ct_vol->pix_spacing[0]);
                ct_xyz[1] = (double) (ct_vol->offset[1] + ct_ijk[1] * ct_vol->pix_spacing[1]);
                ct_xyz[2] = (double) (ct_vol->offset[2] + ct_ijk[2] * ct_vol->pix_spacing[2]);

#if defined (DOSE_DIRECT)
                dose = dose_direct (ct_xyz,         /* voxel to dose */
                                    depth_offset,   /* ap to ct_vol dist */
                                    depth_vol,      /* radiographic depths */
                                    pep,            /* proton energy profile */
                                    pmat,           /* projection matrix */
                                    ires,           /* ray cast resolution */
                                    ul_room,        /* aperture upper-left */
                                    incr_r,         /* 3D ray row increment */
                                    incr_c,         /* 3D ray col increment */
                                    options);       /* options->ray_step */
#endif


#if defined (DOSE_GAUSS)
                dose = dose_scatter (ct_xyz,        /* voxel to dose */
                                     depth_offset,  /* ap to ct_vol dist */
                                     depth_vol,     /* radiographic depths */
                                     pep,           /* proton energy profile */
                                     pmat,          /* projection matrix */
                                     ires,          /* ray cast resolution */
                                     ul_room,       /* aperture upper-left */
                                     incr_r,        /* 3D ray row increment */
                                     incr_c,        /* 3D ray col increment */
                                     prt,           /* x-dir in ap plane uv */
                                     pdn,           /* y-dir in ap plane uv */
                                     options);      /* options->ray_step */
#endif

                /* Insert the dose into the dose volume */
                idx = INDEX_OF (ct_ijk, dose_vol->dim);
                dose_img[idx] = dose;

#if PROGRESS
                /* Report Progress -- slows things down (terribly)*/
                printf (" [%3i%%] %i voxels remaining                \r",
                    (int)floorf((((float)idx/(float)ct_vol->npix)*100.0f)),
                    ct_vol->npix - idx);
                fflush (stdout);
#endif
            }
        }
    }
    printf ("\n");


    free (pep);
    volume_destroy (depth_vol);
}
