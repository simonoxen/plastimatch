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
//#define DOSE_DIRECT
#define DOSE_GAUSS

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

/* This function is used to rotate a
 * point about a ray in an orbit
 * perpendicular to the ray.  It is assumed
 * that the arbitrary axis of rotation (ray)
 * originates at the Cartesian origin.
 */
static void
rotate_point_3D (
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

static inline void
display_progress (
    float is,
    float of
) 
{
    printf (" [%3i%%]\b\b\b\b\b\b\b",
           (int)floorf((is/of)*100.0f));
    fflush (stdout);
}

static void 
debug_voxel (double r,             /* current radius */
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

    fprintf (fp, "r,t: [%3.3f %1.3f] \t rgdepth: %2.3f \t d: %4.1f \t dist: %3.3f \t w: %1.4f \t d0: %4.3f \t sigma: %1.4f\t dose: %4.4f\n",
            r, t, rgdepth, d, vec3_dist(scatter_xyz, ct_xyz), w, d0, sigma, dose);

    fclose (fp);
}
#if defined (commentout)
#endif

/* This needs to be filled in with the
 * actual Highland approximation.
 *
 * Right now this is just a normalization
 */
static double
highland (double rgdepth,
          Proton_Energy_Profile* pep
)
{
    return 3.0*(rgdepth - pep->depth[0]) / (pep->dmax - pep->depth[0]);
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

    if (dist < 0) {
        return d_img[0];
    }

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
    double dist, rgdepth;

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
    rgdepth = rgdepth_lookup (depth_vol,
                               ap_x, ap_y,
                               dist,
                               options->ray_step);

    return rgdepth;
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
    double rgdepth;
    double dose;

    rgdepth = get_rgdepth (ct_xyz,         /* voxel to dose */
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
    if (rgdepth < 0.0f) {
        return 0.0f;
    }

    /* Lookup the dose at this radiographic depth */
    dose = energy_lookup (rgdepth, pep);
    
    return dose;

}

static double
dose_scatter (
    double* ct_xyz,
    int* ct_ijk,            // DEBUG
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

    /* Accounts for 1st order small angle scattering */

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
    double w, d0;   // debug

#if defined (commentout)
    int debug = 0;

//    int watch_ijk[3] = {0, 255, 67};  // entry
    int watch_ijk[3] = {134, 256, 67};  // bragg peak
//    int watch_ijk[3] = {23, 255, 67};    // "stem"

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
    rgdepth = get_rgdepth (ct_xyz, depth_offset,
                            depth_vol, pep,
                            pmat, ires, ap_ul,
                            incr_r, incr_c,
                            options);

    /* If the voxel was not hit *directly* by the beam, there is still a
     * chance that it was hit by scatterers generated by a neighbor who
     * *was* * hit directly by the beam.  As a result, we cannot obtain
     * a resonable estimate, so we assume the largest scattering radius.
     */
    if (rgdepth < 0.0) {
        if (options->detail == 0) {
            rgdepth = pep->dmax;
        } else if (options->detail == 1) {
            /* User wants to ignore "scatter only" dose */
            return 0.0f;
        } else {
            rgdepth = pep->dmax;
        }
    }

    sigma = highland (rgdepth, pep);
    r_max = 3.0*sigma;

    r_step = 1.00;          // mm
    t_step = M_PI / 8.0f;   // radians

    /* Step radius */
    for (r = 0; r < r_max; r += r_step) {
        vec3_copy (sp_pos, ct_xyz);
        vec3_scale3 (tmp, pdn, r);
        vec3_add2 (sp_pos, tmp);

        /* Step angle */
        for (t = 0.0f; t < 2.0*M_PI; t += t_step) {

            rotate_point_3D (scatter_xyz, sp_pos, t, ct_xyz);

            /* neighbor (or self) hit by proton beam? */
            rgdepth = get_rgdepth (scatter_xyz,
                                    depth_offset,
                                    depth_vol, pep,
                                    pmat, ires, ap_ul,
                                    incr_r, incr_c,
                                    options);

            if (rgdepth < 0.0f) {
                continue;
            } else {
                d = energy_lookup (rgdepth, pep);
//                d0 = d; // DEBUG
            }

            vec3_sub3 (sctoct, scatter_xyz, ct_xyz);

            proj_xy[0] = vec3_dot (sctoct, prt);
            proj_xy[1] = vec3_dot (sctoct, pdn);

            sigma = highland (rgdepth, pep);

            /* weight by gaussian kernel */
            w = gaus_kernel (proj_xy, ct_xyz, sigma);
            d *= w;

            /* Add to total dose for our target voxel */
            dose += d;

#if defined (commentout)
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
    double ct_xyz[4];
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
    int idx;


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
                
                /* Transform vol index into space coords */
                ct_xyz[0] = (double) (ct_vol->offset[0] + ct_ijk[0] * ct_vol->pix_spacing[0]);
                ct_xyz[1] = (double) (ct_vol->offset[1] + ct_ijk[1] * ct_vol->pix_spacing[1]);
                ct_xyz[2] = (double) (ct_vol->offset[2] + ct_ijk[2] * ct_vol->pix_spacing[2]);
                ct_xyz[3] = (double) 1.0;

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
                                     ct_ijk,
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

            }
        }
        display_progress ((float)idx, (float)ct_vol->npix);
    }

    free (pep);
    volume_destroy (depth_vol);
}
