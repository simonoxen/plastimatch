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
#include "plmregister_config.h"
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

#include "bspline.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_interpolate.h"
#include "bspline_landmarks.h"
#include "bspline_mi.h"
#include "bspline_mi_hist.h"
#include "bspline_mse.h"
#include "bspline_optimize.h"
#include "bspline_parms.h"
#include "bspline_regularize.h"
#include "bspline_state.h"
#include "bspline_xform.h"
#include "delayload.h"
#include "file_util.h"
#include "interpolate_macros.h"
#include "logfile.h"
#include "plm_math.h"
#include "string_util.h"
#include "volume.h"
#include "volume_macros.h"

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



/* -----------------------------------------------------------------------
   Debugging routines
   ----------------------------------------------------------------------- */
void
dump_gradient (Bspline_xform* bxf, Bspline_score* ssd, const char* fn)
{
    int i;
    FILE* fp;

    make_parent_directories (fn);
    fp = fopen (fn, "wb");
    for (i = 0; i < bxf->num_coeff; i++) {
        fprintf (fp, "%20.20f\n", ssd->grad[i]);
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
    logfile_printf ("         "
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
            bst->mi_hist->dump_hist (bst->it, fn);
        }
    }
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
    Reg_parms* reg_parms = parms->reg_parms;
    Bspline_landmarks* blm = parms->blm;

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
        } else {
            logfile_printf ("\n");
        }
    }
}


void
bspline_score (Bspline_optimize *bod)
{
    Bspline_parms *parms = bod->get_bspline_parms ();
    Bspline_state *bst = bod->get_bspline_state ();
    Bspline_xform *bxf = bod->get_bspline_xform ();

    Reg_parms* reg_parms = parms->reg_parms;
    Bspline_landmarks* blm = parms->blm;

    Volume* fixed_roi  = parms->fixed_roi;
    Volume* moving_roi = parms->moving_roi;
    bool have_roi = fixed_roi || moving_roi;
    bool have_histogram_minmax_val=(parms->mi_fixed_image_minVal!=0)||(parms->mi_fixed_image_maxVal!=0)||(parms->mi_moving_image_minVal!=0)||(parms->mi_moving_image_maxVal!=0);

    /* Zero out the score for this iteration */
    bst->ssd.reset_score ();

    /* CPU Implementations */
    if (parms->threading == BTHR_CPU) {
            
        /* Metric: Mean Squared Error */
        if (parms->metric == BMET_MSE && have_roi) {
            bspline_score_i_mse (bod);
        }
        else if (parms->metric == BMET_MSE) {
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
            case 'k':
                bspline_score_k_mse (bod);
                break;
            case 'l':
                bspline_score_l_mse (bod);
                break;
            case 'm':
                bspline_score_m_mse (bod);
                break;
            case 'n':
                bspline_score_n_mse (bod);
                break;
            default:
#if (OPENMP_FOUND)
                bspline_score_g_mse (bod);
#else
                bspline_score_h_mse (bod);
#endif
                break;
            }
        } /* end MSE */

        /* Metric: Mutual Information with roi or intensity min/max values*/
        else if (parms->metric == BMET_MI && (have_roi || have_histogram_minmax_val))
        {
            switch (parms->implementation) {
            case 'c':
                bspline_score_c_mi (bod);
                break;
#if (OPENMP_FOUND)
            case 'd':
            case 'e':
            case 'f':
            case 'g':
            case 'h':
            case 'i':
                bspline_score_h_mi (bod);
                break;
#endif
            case 'k':
                bspline_score_k_mi (bod);
                break;
            default:
#if (OPENMP_FOUND)
                bspline_score_h_mi (bod);
#else
                bspline_score_c_mi (bod);
#endif
                break;
            }
        }

        /* Metric: Mutual Information without roi */
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
            case 'i':
                bspline_score_i_mi (bod);
                break;
#endif
            case 'k':
                bspline_score_k_mi (bod);
                break;
            default:
#if (OPENMP_FOUND)
                bspline_score_g_mi (bod);
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
            LOAD_LIBRARY_SAFE (libplmregistercuda);
            LOAD_SYMBOL (CUDA_bspline_mse_j, libplmregistercuda);

            switch (parms->implementation) {
            case 'j':
                CUDA_bspline_mse_j (bod);
                break;
            default:
                CUDA_bspline_mse_j (bod);
                break;
            }

            /* Unload plugin when done */
            UNLOAD_LIBRARY (libplmregistercuda);
        } /* end MSE */

        /* Metric: Mutual Information */
        else if (parms->metric == BMET_MI) {

            /* Be sure we loaded the CUDA plugin */
            LOAD_LIBRARY_SAFE (libplmregistercuda);
            LOAD_SYMBOL (CUDA_bspline_mi_a, libplmregistercuda);

            switch (parms->implementation) {
            case 'a':
                CUDA_bspline_mi_a (bod);
                break;
            default:
                CUDA_bspline_mi_a (bod);
                break;
            }

            UNLOAD_LIBRARY (libplmregistercuda);
        } /* end MI */

    } /* CUDA Implementations */
#endif

    /* Regularize */
    if (reg_parms->lambda > 0.0f) {
        bst->rst.compute_score (&bst->ssd, reg_parms, bxf);
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
