/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
    Proposed variable naming guide:
        Fixed image voxel                (f[3]), fidx <currently (fi,fj,fk),fv>
        Moving image voxel               (m[3]), midx < ditto >
            - what about ROI's              ?
            - what about physical coords    ?
        Tile (fixed)                        (t[3]), tidx <currently p[3]>
        Offset within tile (fixed)          (o[3]), oidx <currently q[3]>
        Control point                       (c[3]), cidx <currently (i,j,k), m>
        Coefficient array                   ?                <currently cidx>
        Multiplier LUT                      qidx
        Index LUT                           pidx

    qlut = Multiplier LUT
    clut = Index LUT
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif

#include "plmbase.h"
#include "plmsys.h"

#include "bspline.h"
#include "bspline_mi.h"
#include "bspline_mse.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_regularize.h"
#include "bspline_landmarks.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"

/* EXTERNAL DEPENDS */
#include "bspline_xform.h"
#include "delayload.h"
#include "interpolate_macros.h"
#include "plm_math.h"
#include "volume_macros.h"

/* -----------------------------------------------------------------------
   Initialization and teardown
   ----------------------------------------------------------------------- */
static void
bspline_cuda_state_create (
    Bspline_state *bst,           /* Modified in routine */
    Bspline_xform* bxf,
    Bspline_parms *parms
)
{
#if (CUDA_FOUND)
    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    Dev_Pointers_Bspline* dev_ptrs 
        = (Dev_Pointers_Bspline*) malloc (sizeof (Dev_Pointers_Bspline));

    bst->dev_ptrs = dev_ptrs;
    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {
        /* Be sure we loaded the CUDA plugin */
        LOAD_LIBRARY_SAFE (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mse_init_j, libplmcuda);

        switch (parms->implementation) {
        case 'j':
        case '\0':   /* Default */
            CUDA_bspline_mse_init_j (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        default:
            printf ("Warning: option -f %c unavailble.  Switching to -f j\n",
                parms->implementation);
            CUDA_bspline_mse_init_j (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        }

        UNLOAD_LIBRARY (libplmcuda);
    } 
    else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {

        /* Be sure we loaded the CUDA plugin */
        LOAD_LIBRARY_SAFE (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mi_init_a, libplmcuda);

        switch (parms->implementation) {
        case 'a':
            CUDA_bspline_mi_init_a (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        default:
            printf ("Warning: option -f %c unavailble.  Defaulting to -f a\n",
                parms->implementation);
            CUDA_bspline_mi_init_a (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        }

        UNLOAD_LIBRARY (libplmcuda);
    }
    else {
        printf ("No cuda initialization performed.\n");
    }
#endif
}

Bspline_state *
bspline_state_create (
    Bspline_xform *bxf, 
    Bspline_parms *parms
)
{
    Bspline_state *bst = (Bspline_state*) malloc (sizeof (Bspline_state));
    Reg_parms* reg_parms = &parms->reg_parms;
    Bspline_regularize_state* rst = &bst->rst;
    Bspline_landmarks* blm = &parms->blm;

    memset (bst, 0, sizeof (Bspline_state));
    bst->ssd.grad = (float*) malloc (bxf->num_coeff * sizeof(float));
    memset (bst->ssd.grad, 0, bxf->num_coeff * sizeof(float));

    bspline_cuda_state_create (bst, bxf, parms);

    if (reg_parms->lambda > 0.0f) {
        rst->fixed = parms->fixed;
        rst->moving = parms->moving;
        bspline_regularize_initialize (reg_parms, rst, bxf);
    }

    /* JAS Fix 2011.09.14
     *   The MI algorithm will get stuck for a set of coefficients all equaling
     *   zero due to the method we use to compute the cost function gradient.
     *   However, it is possible we could be inheriting coefficients from a
     *   prior stage, so we must check for inherited coefficients before
     *   applying an initial offset to the coefficient array. */
    if (parms->metric == BMET_MI) {
        bool first_iteration = true;

        for (int i=0; i<bxf->num_coeff; i++) {
            if (bxf->coeff[i] != 0.0f) {
                first_iteration = false;
                break;
            }
        }

        if (first_iteration) {
            printf ("Intializing 1st MI Stage\n");
            for (int i = 0; i < bxf->num_coeff; i++) {
                bxf->coeff[i] = 0.01f;
            }
        }
    }

    /* Landmarks */
    blm->initialize (bxf);

    return bst;
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: calc_offsets()
//
// This function accepts the number or voxels per control region
// and the dimensions of the control grid to generate where the linear
// memory offsets lie for the beginning of each tile in a 32-byte
// aligned tile-major data organization scheme (such as that which
// is produced by kernel_row_to_tile_major().
//
// Author: James Shackleford
// Data: July 30th, 2009
////////////////////////////////////////////////////////////////////////////////
int* calc_offsets(int* tile_dims, int* cdims)
{
    int vox_per_tile = (tile_dims[0] * tile_dims[1] * tile_dims[2]);
    int pad = 32 - (vox_per_tile % 32);
    int num_tiles = (cdims[0]-3)*(cdims[1]-3)*(cdims[2]-3);

    int* output = (int*)malloc(num_tiles*sizeof(int));

    int i;
    for(i = 0; i < num_tiles; i++)
        output[i] = (vox_per_tile + pad) * i;

    return output;
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: find_knots()
//
// This function takes a tile index as an input and generates
// the indicies of the 64 control knots that it affects.
//
// Returns:
//    knots[idx] - idx is [0,63] and knots[idx] = the linear knot index
//                 of affected control knot # idx within the entire control
//                 knot grid.
//
//    tile_pos[
//
// Author: James Shackleford
// Data: July 13th, 2009
////////////////////////////////////////////////////////////////////////////////
void find_knots (
    plm_long* knots, 
    plm_long tile_num, 
    plm_long* rdims, 
    plm_long* cdims
) {
    int tile_loc[3];
    int i, j, k;
    int idx = 0;
    int num_tiles_x = cdims[0] - 3;
    int num_tiles_y = cdims[1] - 3;
    int num_tiles_z = cdims[2] - 3;
        
    // First get the [x,y,z] coordinate of
    // the tile in the control grid.
    tile_loc[0] = tile_num % num_tiles_x;
    tile_loc[1] = ((tile_num - tile_loc[0]) / num_tiles_x) % num_tiles_y;
    tile_loc[2] = ((((tile_num - tile_loc[0]) / num_tiles_x) / num_tiles_y) % num_tiles_z);
    /*
      tile_loc[0] = tile_num % rdims[0];
      tile_loc[1] = ((tile_num - tile_loc[0]) / rdims[0]) % rdims[1];
      tile_loc[2] = ((((tile_num - tile_loc[0]) / rdims[0]) / rdims[1]) % rdims[2]);
    */

    // Tiles do not start on the edges of the grid, so we
    // push them to the center of the control grid.
    tile_loc[0]++;
    tile_loc[1]++;
    tile_loc[2]++;

    // Find 64 knots' [x,y,z] coordinates
    // and convert into a linear knot index
    for (k = -1; k < 3; k++)
        for (j = -1; j < 3; j++)
            for (i = -1; i < 3; i++)
            {
                knots[idx++] = (cdims[0]*cdims[1]*(tile_loc[2]+k)) + (cdims[0]*(tile_loc[1]+j)) + (tile_loc[0]+i);
            }

}

static void
float_to_plm_long_clamp (plm_long* p, float* f, const plm_long* max)
{
    float p_f[3];
    p_f[0] = f[0];
    p_f[1] = f[1];
    p_f[2] = f[2];

    /* x */
    if (p_f[0] < 0.f) {
        p[0] = 0;
    }
    else if (p_f[0] >= max[0]) {
        p[0] = max[0] - 1;
    }
    else {
        p[0] = FLOOR_PLM_LONG (p_f[0]);
    }

    /* y */
    if (p_f[1] < 0.f) {
        p[1] = 0;
    }
    else if (p_f[1] >= max[1]) {
        p[1] = max[1] - 1;
    }
    else {
        p[1] = FLOOR_PLM_LONG (p_f[1]);
    }

    /* z */
    if (p_f[2] < 0.f) {
        p[2] = 0;
    }
    else if (p_f[2] >= max[2]) {
        p[2] = max[2] - 1;
    }
    else {
        p[2] = FLOOR_PLM_LONG (p_f[2]);
    }
}

int
inside_mask (float* xyz, const Volume* mask)
{
    float p_f[3];
    float tmp[3];
    plm_long p[3];

    tmp[0] = xyz[0] - mask->offset[0];
    tmp[1] = xyz[1] - mask->offset[1];
    tmp[2] = xyz[2] - mask->offset[2];

    p_f[0] = PROJECT_X (tmp, mask->proj);
    p_f[1] = PROJECT_Y (tmp, mask->proj);
    p_f[2] = PROJECT_Z (tmp, mask->proj);

    float_to_plm_long_clamp (p, p_f, mask->dim);

    unsigned char *m = (unsigned char*)mask->img;
    plm_long i = volume_index (mask->dim, p);

    /* 0 outside mask, 1 inside */
    return (int)m[i];
}


/* -----------------------------------------------------------------------
   Debugging routines
   ----------------------------------------------------------------------- */
void
dump_gradient (Bspline_xform* bxf, Bspline_score* ssd, const char* fn)
{
    int i;
    FILE* fp;

    make_directory_recursive (fn);
    fp = fopen (fn, "wb");
    for (i = 0; i < bxf->num_coeff; i++) {
        fprintf (fp, "%20.20f\n", ssd->grad[i]);
    }
    fclose (fp);
}

void
dump_hist (BSPLINE_MI_Hist* mi_hist, int it, const std::string& prefix)
{
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    plm_long i, j, v;
    FILE *fp;
    //char fn[_MAX_PATH];
    std::string fn;
    char buf[_MAX_PATH];

    sprintf (buf, "hist_fix_%02d.csv", it);
    fn = prefix + buf;
    make_directory_recursive (fn.c_str());
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;
    for (plm_long i = 0; i < mi_hist->fixed.bins; i++) {
        fprintf (fp, "%u %f\n", (unsigned int) i, f_hist[i]);
    }
    fclose (fp);

    sprintf (buf, "hist_mov_%02d.csv", it);
    fn = prefix + buf;
    make_directory_recursive (fn.c_str());
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;
    for (i = 0; i < mi_hist->moving.bins; i++) {
        fprintf (fp, "%u %f\n", (unsigned int) i, m_hist[i]);
    }
    fclose (fp);

    sprintf (buf, "hist_jnt_%02d.csv", it);
    fn = prefix + buf;
    make_directory_recursive (fn.c_str());
    fp = fopen (fn.c_str(), "wb");
    if (!fp) return;
    for (i = 0, v = 0; i < mi_hist->fixed.bins; i++) {
        for (j = 0; j < mi_hist->moving.bins; j++, v++) {
            if (j_hist[v] > 0) {
                fprintf (fp, "%u %u %u %g\n", (unsigned int) i, 
                    (unsigned int) j, (unsigned int) v, j_hist[v]);
            }
        }
    }
    fclose (fp);
}

void
bspline_display_coeff_stats (Bspline_xform* bxf)
{
    float cf_min, cf_avg, cf_max;
    int i;

    cf_avg = 0.0;
    cf_min = cf_max = bxf->coeff[0];
    for (i = 0; i < bxf->num_coeff; i++) {
        cf_avg += bxf->coeff[i];
        if (cf_min > bxf->coeff[i]) cf_min = bxf->coeff[i];
        if (cf_max < bxf->coeff[i]) cf_max = bxf->coeff[i];
    }
    logfile_printf ("                    "
                    "CMIN %6.2f CAVG %6.2f CMAX %6.2f\n", 
                    cf_min, cf_avg / bxf->num_coeff, cf_max);
}

void
bspline_save_debug_state (
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform *bxf
)
{
    if (parms->debug) {
        std::string fn;
        char buf[1024];

        if (parms->metric == BMET_MI) {
            sprintf (buf, "%02d_grad_mi_%03d_%03d.txt", 
                parms->debug_stage, bst->it, bst->feval);
        } else {
            sprintf (buf, "%02d_grad_mse_%03d_%03d.txt", 
                parms->debug_stage, bst->it, bst->feval);
        }
        fn = parms->debug_dir + "/" + buf;
        dump_gradient (bxf, &bst->ssd, fn.c_str());

        sprintf (buf, "%02d_coeff_%03d_%03d.txt", 
            parms->debug_stage, bst->it, bst->feval);
        fn = parms->debug_dir + "/" + buf;
        bspline_xform_save (bxf, fn.c_str());

        if (parms->metric == BMET_MI) {
            sprintf (buf, "%02d_", parms->debug_stage);
            fn = parms->debug_dir + "/" + buf;
            dump_hist (&parms->mi_hist, bst->it, fn);
        }
    }
}

void
bspline_parms_free (Bspline_parms* parms)
{
    if (parms->mi_hist.fixed.type == HIST_VOPT) {
        free (parms->mi_hist.fixed.key_lut);
    }
    if (parms->mi_hist.moving.type == HIST_VOPT) {
        free (parms->mi_hist.moving.key_lut);
    }

    if (parms->mi_hist.j_hist) {
        free (parms->mi_hist.f_hist);
        free (parms->mi_hist.m_hist);
        free (parms->mi_hist.j_hist);
    }
}

void
bspline_state_destroy (
    Bspline_state *bst,
    Bspline_parms *parms, 
    Bspline_xform *bxf
)
{
    Reg_parms* reg_parms = &parms->reg_parms;

    if (bst->ssd.grad) {
        free (bst->ssd.grad);
    }

    if (reg_parms->lambda > 0.0f) {
        bspline_regularize_destroy (reg_parms, &bst->rst, bxf);
    }

#if (CUDA_FOUND)
    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {
        LOAD_LIBRARY_SAFE (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mse_cleanup_j, libplmcuda);
        CUDA_bspline_mse_cleanup_j ((Dev_Pointers_Bspline *) bst->dev_ptrs, fixed, moving, moving_grad);
        UNLOAD_LIBRARY (libplmcuda);
    }
    else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {
        LOAD_LIBRARY_SAFE (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mi_cleanup_a, libplmcuda);
        CUDA_bspline_mi_cleanup_a ((Dev_Pointers_Bspline *) bst->dev_ptrs, fixed, moving, moving_grad);
        UNLOAD_LIBRARY (libplmcuda);
    }
#endif

    free (bst);
}

Volume*
bspline_compute_vf (const Bspline_xform* bxf)
{
    Volume* vf = new Volume (
        bxf->img_dim, bxf->img_origin, 
        bxf->img_spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 3
    );
    bspline_interpolate_vf (vf, bxf);

    return vf;
}

void
bspline_update_sets (float* sets_x, float* sets_y, float* sets_z,
    int qidx, float* dc_dv, Bspline_xform* bxf)
{
    int sidx;   // set index

    /* Initialize q_lut */
    float* q_lut = &bxf->q_lut[64*qidx];

    /* Condense dc_dv & commit to sets for tile */
    for (sidx=0; sidx<64; sidx++) {
        sets_x[sidx] += dc_dv[0] * q_lut[sidx];
        sets_y[sidx] += dc_dv[1] * q_lut[sidx];
        sets_z[sidx] += dc_dv[2] * q_lut[sidx];
    }
}

void
bspline_update_sets_b (float* sets_x, float* sets_y, float* sets_z,
    plm_long *q, float* dc_dv, Bspline_xform* bxf)
{
    int i,j,k,m;
    float A,B,C;

    /* Initialize b_luts */
    float* bx_lut = &bxf->bx_lut[q[0]*4];
    float* by_lut = &bxf->by_lut[q[1]*4];
    float* bz_lut = &bxf->bz_lut[q[2]*4];

    /* Condense dc_dv & commit to sets for tile */
    m=0;
    for (k=0; k<4; k++) {
        C = bz_lut[k];
        for (j=0; j<4; j++) {
            B = by_lut[j] * C;
            for (i=0; i<4; i++) {
                A = bx_lut[i] * B;
                sets_x[m] += dc_dv[0] * A;
                sets_y[m] += dc_dv[1] * A;
                sets_z[m] += dc_dv[2] * A;
                m++;
            }
        }
    }
}


void
bspline_sort_sets (float* cond_x, float* cond_y, float* cond_z,
    float* sets_x, float* sets_y, float* sets_z,
    plm_long pidx, Bspline_xform* bxf)
{
    int sidx, kidx;
    plm_long* k_lut = (plm_long*) malloc (64*sizeof(plm_long));

    /* Generate the knot index lut */
    find_knots (k_lut, pidx, bxf->rdims, bxf->cdims);

    /* Rackem' Up */
    for (sidx=0; sidx<64; sidx++) {
        kidx = k_lut[sidx];

        cond_x[ (64*kidx) + sidx ] = sets_x[sidx];
        cond_y[ (64*kidx) + sidx ] = sets_y[sidx];
        cond_z[ (64*kidx) + sidx ] = sets_z[sidx];
    }

    free (k_lut);
}

void
bspline_update_grad (
    Bspline_state *bst, 
    Bspline_xform* bxf, 
    plm_long p[3], plm_long qidx, float dc_dv[3])
{
    Bspline_score* ssd = &bst->ssd;
    plm_long i, j, k, m;
    plm_long cidx;
    float* q_lut = &bxf->q_lut[qidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
        for (j = 0; j < 4; j++) {
            for (i = 0; i < 4; i++) {
                cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
                        + (p[1] + j) * bxf->cdims[0]
                        + (p[0] + i);
                cidx = cidx * 3;
                ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
                ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
                ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
                m ++;
            }
        }
    }
}

void
bspline_update_grad_b (
    Bspline_score* bscore,
    const Bspline_xform* bxf, 
    plm_long pidx, 
    plm_long qidx, 
    const float dc_dv[3])
{
    plm_long i, j, k, m;
    plm_long cidx;
    float* q_lut = &bxf->q_lut[qidx*64];
    plm_long* c_lut = &bxf->c_lut[pidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
        for (j = 0; j < 4; j++) {
            for (i = 0; i < 4; i++) {
                cidx = 3 * c_lut[m];
                bscore->grad[cidx+0] += dc_dv[0] * q_lut[m];
                bscore->grad[cidx+1] += dc_dv[1] * q_lut[m];
                bscore->grad[cidx+2] += dc_dv[2] * q_lut[m];
                m ++;
            }
        }
    }
}

void
bspline_make_grad (float* cond_x, float* cond_y, float* cond_z,
                   Bspline_xform* bxf, Bspline_score* ssd)
{
    plm_long kidx, sidx;

    for (kidx=0; kidx < (bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2]); kidx++) {
        for (sidx=0; sidx<64; sidx++) {
            ssd->grad[3*kidx + 0] += cond_x[64*kidx + sidx];
            ssd->grad[3*kidx + 1] += cond_y[64*kidx + sidx];
            ssd->grad[3*kidx + 2] += cond_z[64*kidx + sidx];
        }
    }
}

void
report_score (
    Bspline_parms *parms,
    Bspline_xform *bxf, 
    Bspline_state *bst
) 
{
    Bspline_score* ssd = &bst->ssd;
    Reg_parms* reg_parms = &parms->reg_parms;
    Bspline_landmarks* blm = &parms->blm;

    int i;
    float ssd_grad_norm, ssd_grad_mean;

    /* Normalize gradient */
    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
        ssd_grad_mean += bst->ssd.grad[i];
        ssd_grad_norm += fabs (bst->ssd.grad[i]);
    }

    /* First line, part 1 - iterations */
    logfile_printf ("[%2d,%3d] ", bst->it, bst->feval);
    /* First line, part 2 - score 
       JAS 04.19.2010 MI scores are between 0 and 1
       The extra decimal point resolution helps in seeing
       if the optimizer is performing adequately. */
    if (reg_parms->lambda > 0 || blm->num_landmarks > 0) {
        logfile_printf ("SCORE ");
    } else if (parms->metric == BMET_MI) {
        logfile_printf ("MI  ");
    } else {
        logfile_printf ("MSE ");
    }
    if (parms->metric == BMET_MI) {
        logfile_printf ("%1.8f ", ssd->score);
    } else {
        logfile_printf ("%9.3f ", ssd->score);
    }
    /* First line, part 3 - misc stats */
    logfile_printf (
            "NV %6d GM %9.3f GN %9.3f [ %9.3f s ]\n",
            ssd->num_vox, ssd_grad_mean, ssd_grad_norm, 
            ssd->time_smetric + ssd->time_rmetric);

    /* Second line - extra stats if regularization is enabled */
    if (reg_parms->lambda > 0 || blm->num_landmarks > 0) {
        /* Part 1 - similarity metric */
        logfile_printf (
            "         %s %9.3f ", 
            (parms->metric == BMET_MI) ? "MI   " : "MSE  ", ssd->smetric);
        /* Part 2 - regularization metric */
        if (reg_parms->lambda > 0) {
            logfile_printf ("RM %9.3f ", 
                reg_parms->lambda * bst->ssd.rmetric);
        }
        /* Part 3 - landmark metric */
        if (blm->num_landmarks > 0) {
            logfile_printf ("LM %9.3f ", 
                blm->landmark_stiffness * bst->ssd.lmetric);
        }
        /* Part 4 - timing */
        if (reg_parms->lambda > 0) {
            logfile_printf ("[ %9.3f | %9.3f ]\n", 
                ssd->time_smetric, ssd->time_rmetric);
        }
    }
}

/* -----------------------------------------------------------------------
   Macros
   ----------------------------------------------------------------------- */
/* Find location and index of corresponding voxel in moving image.  
   Return 1 if corresponding voxel lies within the moving image, 
   return 0 if outside the moving image.  */
int
bspline_find_correspondence 
(
 float *mxyz,             /* Output: xyz coordinates in moving image (mm) */
 float *mijk,             /* Output: ijk indices in moving image (vox) */
 const float *fxyz,       /* Input:  xyz coordinates in fixed image (mm) */
 const float *dxyz,       /* Input:  displacement from fixed to moving (mm) */
 const Volume *moving     /* Input:  moving image */
 )
{
    mxyz[0] = fxyz[0] + dxyz[0];
    mijk[0] = (mxyz[0] - moving->offset[0]) / moving->spacing[0];
    if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) return 0;

    mxyz[1] = fxyz[1] + dxyz[1];
    mijk[1] = (mxyz[1] - moving->offset[1]) / moving->spacing[1];
    if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) return 0;

    mxyz[2] = fxyz[2] + dxyz[2];
    mijk[2] = (mxyz[2] - moving->offset[2]) / moving->spacing[2];
    if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) return 0;

    return 1;
}

/* Find location and index of corresponding voxel in moving image.
 * This version takes direction cosines into consideration
   Return 1 if corresponding voxel lies within the moving image, 
   return 0 if outside the moving image.  */
int
bspline_find_correspondence_dcos
(
 float *mxyz,             /* Output: xyz coordinates in moving image (mm) */
 float *mijk,             /* Output: ijk indices in moving image (vox) */
 const float *fxyz,       /* Input:  xyz coordinates in fixed image (mm) */
 const float *dxyz,       /* Input:  displacement from fixed to moving (mm) */
 const Volume *moving     /* Input:  moving image */
 )
{
    float tmp[3];

    mxyz[0] = fxyz[0] + dxyz[0];
    mxyz[1] = fxyz[1] + dxyz[1];
    mxyz[2] = fxyz[2] + dxyz[2];

    tmp[0] = mxyz[0] - moving->offset[0];
    tmp[1] = mxyz[1] - moving->offset[1];
    tmp[2] = mxyz[2] - moving->offset[2];

    mijk[0] = PROJECT_X (tmp, moving->proj);
    mijk[1] = PROJECT_Y (tmp, moving->proj);
    mijk[2] = PROJECT_Z (tmp, moving->proj);

    if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) return 0;
    if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) return 0;
    if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) return 0;

    return 1;
}

/* Find location and index of corresponding voxel in moving image.
 * This version takes direction cosines and true masking into consideration
   Return 1 if corresponding voxel lies within the moving image, 
   return 0 if outside the moving image.  */
int
bspline_find_correspondence_dcos_mask
(
 float *mxyz,               /* Output: xyz coordinates in moving image (mm) */
 float *mijk,               /* Output: ijk indices in moving image (vox) */
 const float *fxyz,         /* Input:  xyz coordinates in fixed image (mm) */
 const float *dxyz,         /* Input:  displacement from fixed to moving (mm) */
 const Volume *moving,      /* Input:  moving image */
 const Volume *moving_mask  /* Input:  moving image mask */
 )
{
    float tmp[3];

    mxyz[0] = fxyz[0] + dxyz[0];
    mxyz[1] = fxyz[1] + dxyz[1];
    mxyz[2] = fxyz[2] + dxyz[2];

    tmp[0] = mxyz[0] - moving->offset[0];
    tmp[1] = mxyz[1] - moving->offset[1];
    tmp[2] = mxyz[2] - moving->offset[2];

    mijk[0] = PROJECT_X (tmp, moving->proj);
    mijk[1] = PROJECT_Y (tmp, moving->proj);
    mijk[2] = PROJECT_Z (tmp, moving->proj);

    if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) return 0;
    if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) return 0;
    if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) return 0;

    if (moving_mask) {
        return inside_mask (mxyz, moving_mask);
    }

    return 1;
}

void
bspline_score (Bspline_optimize_data *bod)
{
    Bspline_parms *parms = bod->parms;
    Bspline_state *bst   = bod->bst;
    Bspline_xform *bxf   = bod->bxf;

    Reg_parms* reg_parms = &parms->reg_parms;
    Bspline_landmarks* blm = &parms->blm;

    /* CPU Implementations */
    if (parms->threading == BTHR_CPU) {
            
        /* Metric: Mean Squared Error */
        if (parms->metric == BMET_MSE) {
            switch (parms->implementation) {
            case 'c':
                bspline_score_c_mse (bod);
                break;
            case 'g':
                bspline_score_g_mse (bod);
                break;
            case 'h':
                bspline_score_h_mse (bod);
                break;
            case 'i':
                bspline_score_i_mse (bod);
                break;
            default:
                bspline_score_g_mse (bod);
                break;
            }
        } /* end MSE */

        /* Metric: Mutual Information */
        else if (parms->metric == BMET_MI) {
            switch (parms->implementation) {
            case 'c':
                bspline_score_c_mi (bod);
                break;
#if (OPENMP_FOUND)
            case 'd':
                bspline_score_d_mi (bod);
                break;
            case 'e':
                bspline_score_e_mi (bod);
                break;
            case 'f':
                bspline_score_f_mi (bod);
                break;
            case 'g':
                bspline_score_g_mi (bod);
                break;
            case 'h':
                bspline_score_h_mi (bod);
                break;
#endif
            default:
#if (OPENMP_FOUND)
                bspline_score_d_mi (bod);
#else
                bspline_score_c_mi (bod);
#endif
                break;
            }
        } /* end MI */

    } /* end CPU Implementations */


    /* CUDA Implementations */
#if (CUDA_FOUND)
    else if (parms->threading == BTHR_CUDA) {
            
        /* Metric: Mean Squared Error */
        if (parms->metric == BMET_MSE) {

            /* Be sure we loaded the CUDA plugin */
            LOAD_LIBRARY_SAFE (libplmcuda);
            LOAD_SYMBOL (CUDA_bspline_mse_j, libplmcuda);

            switch (parms->implementation) {
            case 'j':
                CUDA_bspline_mse_j (bod);
                break;
            default:
                CUDA_bspline_mse_j (bod);
                break;
            }

            /* Unload plugin when done */
            UNLOAD_LIBRARY (libplmcuda);
        } /* end MSE */

        /* Metric: Mutual Information */
        else if (parms->metric == BMET_MI) {

            /* Be sure we loaded the CUDA plugin */
            LOAD_LIBRARY_SAFE (libplmcuda);
            LOAD_SYMBOL (CUDA_bspline_mi_a, libplmcuda);

            switch (parms->implementation) {
            case 'a':
                CUDA_bspline_mi_a (bod);
                break;
            default: 
                CUDA_bspline_mi_a (bod);
                break;
            }

            UNLOAD_LIBRARY (libplmcuda);
        } /* end MI */

    } /* CUDA Implementations */
#endif



    /* Regularize */
    if (reg_parms->lambda > 0.0f) {
        bspline_regularize (&bst->ssd, &bst->rst, reg_parms, bxf);
    }

    /* Compute landmark score/gradient to image score/gradient */
    if (blm->num_landmarks > 0) {
        bspline_landmarks_score (parms, bst, bxf);
    }

    /* Compute total score to send of optimizer */
    bst->ssd.score = bst->ssd.smetric + reg_parms->lambda * bst->ssd.rmetric;
    if (blm->num_landmarks > 0) {
        bst->ssd.score += blm->landmark_stiffness * bst->ssd.lmetric;
    }

    /* Report results of this iteration */
    report_score (parms, bxf, bst);
}
