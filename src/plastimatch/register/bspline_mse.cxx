/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif

#include "plmsys.h"

#include "bspline.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_mse.h"
#include "bspline_optimize.h"
#include "bspline_opts.h"
#include "plm_math.h"
#include "volume.h"
#include "volume_macros.h"
#include "bspline_macros.h"

#include "interpolate_macros.h"

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_score_h_mse()
//
// This is a single core CPU implementation of CUDA implementation J.
// The tile "condense" method is demonstrated.
//
// ** This is the fastest know CPU implmentation for single core **
//
// See also:
//   OpenMP implementation of CUDA J: bspline_score_g_mse()
//
// AUTHOR: James A. Shackleford
// DATE: 11.22.2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_score_h_mse (
    Bspline_optimize_data *bod
)
{
    Bspline_parms *parms = bod->parms;
    Bspline_state *bst = bod->bst;
    Bspline_xform *bxf = bod->bxf;

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    Bspline_score* ssd = &bst->ssd;
    double score_tile;

    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;

    plm_long idx_tile;

    Plm_timer *timer = plm_timer_create();

    plm_long cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    int i;

    // Start timing the code
    plm_timer_start (timer);

    // Zero out accumulators
    ssd->smetric = 0;
    ssd->num_vox = 0;
    score_tile = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

    // Serial across tiles
    LOOP_THRU_VOL_TILES (idx_tile, bxf) {
        int rc;

        int ijk_tile[3];
        plm_long ijk_local[3];
        plm_long idx_local;

        float xyz_fixed[3];
        plm_long ijk_fixed[3];
        plm_long idx_fixed;

        float dxyz[3];

        float xyz_moving[3];
        float ijk_moving[3];
        plm_long ijk_moving_floor[3];
        plm_long ijk_moving_round[3];
        plm_long idx_moving_floor;
        plm_long idx_moving_round;

        float li_1[3], li_2[3];
        float m_val, diff;
    
        float dc_dv[3];

        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        // Get tile coordinates from index
        COORDS_FROM_INDEX (ijk_tile, idx_tile, bxf->rdims); 

        // Serial through voxels in tile
        LOOP_THRU_TILE_Z (ijk_local, bxf) {
            LOOP_THRU_TILE_Y (ijk_local, bxf) {
                LOOP_THRU_TILE_X (ijk_local, bxf) {

                    // Construct coordinates into fixed image volume
                    GET_VOL_COORDS (ijk_fixed, ijk_tile, ijk_local, bxf);

                    // Make sure we are inside the region of interest
                    if (ijk_fixed[0] >= bxf->roi_offset[0] + bxf->roi_dim[0])
                        continue;
                    if (ijk_fixed[1] >= bxf->roi_offset[1] + bxf->roi_dim[1])
                        continue;
                    if (ijk_fixed[2] >= bxf->roi_offset[2] + bxf->roi_dim[2])
                        continue;

                    // Compute physical coordinates of fixed image voxel
                    GET_REAL_SPACE_COORDS (xyz_fixed, ijk_fixed, bxf);

                    // Construct the image volume index
                    idx_fixed = volume_index (fixed->dim, ijk_fixed);

                    // Calc. deformation vector (dxyz) for voxel
                    bspline_interp_pix_c (dxyz, bxf, idx_tile, ijk_local);

                    // Calc. moving image coordinate from the deformation vector
                    rc = bspline_find_correspondence (xyz_moving, ijk_moving,
                        xyz_fixed, dxyz, moving);

                    // Return code is 0 if voxel is pushed outside of moving image
                    if (!rc) continue;

                    // Compute linear interpolation fractions
                    li_clamp_3d (
                        ijk_moving,
                        ijk_moving_floor,
                        ijk_moving_round,
                        li_1,
                        li_2,
                        moving
                    );

                    // Find linear indices for moving image
                    idx_moving_floor = volume_index (moving->dim, ijk_moving_floor);
                    idx_moving_round = volume_index (moving->dim, ijk_moving_round);

                    // Calc. moving voxel intensity via linear interpolation
                    LI_VALUE (
                        m_val, 
                        li_1[0], li_2[0],
                        li_1[1], li_2[1],
                        li_1[2], li_2[2],
                        idx_moving_floor,
                        m_img, moving
                    );

                    // Compute intensity difference
                    diff = m_val - f_img[idx_fixed];

                    // Store the score!
                    score_tile += diff * diff;
                    ssd->num_vox++;

                    // Compute dc_dv
                    dc_dv[0] = diff * m_grad[3 * idx_moving_round + 0];
                    dc_dv[1] = diff * m_grad[3 * idx_moving_round + 1];
                    dc_dv[2] = diff * m_grad[3 * idx_moving_round + 2];

                    /* Generate condensed tile */
                    bspline_update_sets_b (sets_x, sets_y, sets_z,
                        ijk_local, dc_dv, bxf);

                } /* LOOP_THRU_TILE_X */
            } /* LOOP_THRU_TILE_Y */
        } /* LOOP_THRU_TILE_Z */
        

        // The tile is now condensed.  Now we will put it in the
        // proper slot within the control point bin that it belong to.
        bspline_sort_sets (
            cond_x, cond_y, cond_z,
            sets_x, sets_y, sets_z,
            idx_tile, bxf
        );


    } /* LOOP_THRU_VOL_TILES */


    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  A single bin summation is
     * dc_dp for the single control point associated with the bin.
     * The number of total bins is equal to the number of control
     * points in the control grid.
     */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

    ssd->smetric = score_tile / ssd->num_vox;

    for (i = 0; i < bxf->num_coeff; i++) {
        ssd->grad[i] = 2 * ssd->grad[i] / ssd->num_vox;
    }

    ssd->time_smetric = plm_timer_report (timer);
    plm_timer_destroy (timer);
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_score_g_mse()
//
// This is a multi-CPU implementation of CUDA implementation J.  OpenMP is
// used.  The tile "condense" method is demonstrated.
//
// ** This is the fastest know CPU implmentation for multi core **
//    (That does not require SSE)
//
// AUTHOR: James A. Shackleford
// DATE: 11.22.2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_score_g_mse (
    Bspline_optimize_data *bod
)
{
    Bspline_parms *parms = bod->parms;
    Bspline_state *bst = bod->bst;
    Bspline_xform *bxf = bod->bxf;

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    Bspline_score* ssd = &bst->ssd;
    double score_tile;

    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;

    int idx_tile;

    Plm_timer* timer = plm_timer_create ();

    plm_long cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    int i;

    // Start timing the code
    plm_timer_start (timer);

    // Zero out accumulators
    int num_vox = 0;
    ssd->smetric = 0;
    score_tile = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

    // Parallel across tiles
#pragma omp parallel for reduction (+:num_vox,score_tile)
    LOOP_THRU_VOL_TILES (idx_tile, bxf) {
        int rc;

        plm_long ijk_tile[3];
        plm_long ijk_local[3];
        plm_long idx_local;

        float xyz_fixed[3];
        plm_long ijk_fixed[3];
        plm_long idx_fixed;

        float dxyz[3];

        float xyz_moving[3];
        float ijk_moving[3];
        plm_long ijk_moving_floor[3];
        plm_long ijk_moving_round[3];
        plm_long idx_moving_floor;
        plm_long idx_moving_round;

        float li_1[3], li_2[3];
        float m_val, diff;
    
        float dc_dv[3];

        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        // Get tile coordinates from index
        COORDS_FROM_INDEX (ijk_tile, idx_tile, bxf->rdims); 

        // Serial through voxels in tile
        LOOP_THRU_TILE_Z (ijk_local, bxf) {
            LOOP_THRU_TILE_Y (ijk_local, bxf) {
                LOOP_THRU_TILE_X (ijk_local, bxf) {

                    // Construct coordinates into fixed image volume
                    GET_VOL_COORDS (ijk_fixed, ijk_tile, ijk_local, bxf);

                    // Make sure we are inside the image volume
                    if (ijk_fixed[0] >= bxf->roi_offset[0] + bxf->roi_dim[0])
                        continue;
                    if (ijk_fixed[1] >= bxf->roi_offset[1] + bxf->roi_dim[1])
                        continue;
                    if (ijk_fixed[2] >= bxf->roi_offset[2] + bxf->roi_dim[2])
                        continue;

                    // Compute physical coordinates of fixed image voxel
                    GET_REAL_SPACE_COORDS (xyz_fixed, ijk_fixed, bxf);
                    
                    // Construct the image volume index
                    idx_fixed = volume_index (fixed->dim, ijk_fixed);

                    // Calc. deformation vector (dxyz) for voxel
                    bspline_interp_pix_c (dxyz, bxf, idx_tile, ijk_local);

                    // Calc. moving image coordinate from the deformation vector
                    rc = bspline_find_correspondence (xyz_moving, ijk_moving,
                        xyz_fixed, dxyz, moving);

                    // Return code is 0 if voxel is pushed outside of 
                    // moving image
                    if (!rc) continue;

                    // Compute linear interpolation fractions
                    li_clamp_3d (
                        ijk_moving,
                        ijk_moving_floor,
                        ijk_moving_round,
                        li_1,
                        li_2,
                        moving
                    );

                    // Find linear indices for moving image
                    idx_moving_floor = volume_index (moving->dim, ijk_moving_floor);
                    idx_moving_round = volume_index (moving->dim, ijk_moving_round);

                    // Calc. moving voxel intensity via linear interpolation
                    LI_VALUE (
                        m_val, 
                        li_1[0], li_2[0],
                        li_1[1], li_2[1],
                        li_1[2], li_2[2],
                        idx_moving_floor,
                        m_img, moving
                    );

                    // Compute intensity difference
                    diff = m_val - f_img[idx_fixed];

                    // Store the score!
                    score_tile += diff * diff;
                    num_vox++;

                    // Compute dc_dv
                    dc_dv[0] = diff * m_grad[3 * idx_moving_round + 0];
                    dc_dv[1] = diff * m_grad[3 * idx_moving_round + 1];
                    dc_dv[2] = diff * m_grad[3 * idx_moving_round + 2];

                    /* Generate condensed tile */
                    bspline_update_sets_b (
                        sets_x, sets_y, sets_z,
                        ijk_local, dc_dv, bxf
                    );

                } /* LOOP_THRU_TILE_X */
            } /* LOOP_THRU_TILE_Y */
        } /* LOOP_THRU_TILE_Z */


        // The tile is now condensed.  Now we will put it in the
        // proper slot within the control point bin that it belong to.
        bspline_sort_sets (
            cond_x, cond_y, cond_z,
            sets_x, sets_y, sets_z,
            idx_tile, bxf
        );

    } /* LOOP_THRU_VOL_TILES */

    ssd->num_vox = num_vox;

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

    ssd->smetric = score_tile / ssd->num_vox;

    for (i = 0; i < bxf->num_coeff; i++) {
        ssd->grad[i] = 2 * ssd->grad[i] / ssd->num_vox;
    }

    /* Save for reporting */
    ssd->time_smetric = plm_timer_report (timer);
    plm_timer_destroy (timer);
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_score_c_mse()
//
// This is the older "fast" single-threaded MSE implementation.
////////////////////////////////////////////////////////////////////////////////
void
bspline_score_c_mse (
    Bspline_optimize_data *bod
)
{
    Bspline_parms *parms = bod->parms;
    Bspline_state *bst = bod->bst;
    Bspline_xform *bxf = bod->bxf;

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    Bspline_score* ssd = &bst->ssd;
    plm_long rijk[3];             /* Indices within fixed image region (vox) */
    plm_long fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];           /* Indices within moving image (vox) */
    float fxyz[3];           /* Position within fixed image (mm) */
    float mxyz[3];           /* Position within moving image (mm) */
    plm_long mijk_f[3], mvf;      /* Floor */
    plm_long mijk_r[3], mvr;      /* Round */
    plm_long p[3];
    plm_long q[3];
    float diff;
    float dc_dv[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    plm_long pidx, qidx;
    float m_val;

    /* GCS: Oct 5, 2009.  We have determined that sequential accumulation
       of the score requires double precision.  However, reduction 
       accumulation does not. */
    double score_acc = 0.;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp = 0;

    if (parms->debug) {
        sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
        fp = fopen (debug_fn, "wb");
    }

    Plm_timer* timer = plm_timer_create ();
    plm_timer_start (timer);

    ssd->num_vox = 0;
    ssd->smetric = 0.0f;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    LOOP_THRU_ROI_Z (rijk, fijk, bxf) {
        p[2] = REGION_INDEX_Z (rijk, bxf);
        q[2] = REGION_OFFSET_Z (rijk, bxf);
        fxyz[2] = GET_REAL_SPACE_COORD_Z (fijk, bxf);

        LOOP_THRU_ROI_Y (rijk, fijk, bxf) {
            p[1] = REGION_INDEX_Y (rijk, bxf);
            q[1] = REGION_OFFSET_Y (rijk, bxf);
            fxyz[1] = GET_REAL_SPACE_COORD_Y (fijk, bxf);

            LOOP_THRU_ROI_X (rijk, fijk, bxf) {
                int rc;
                p[0] = REGION_INDEX_X (rijk, bxf);
                q[0] = REGION_OFFSET_X (rijk, bxf);
                fxyz[0] = GET_REAL_SPACE_COORD_X (fijk, bxf);

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Compute moving image coordinate of fixed image voxel */
                rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
                    dxyz, moving);

                /* If voxel is not inside moving image */
                if (!rc) {
                    continue;
                }

                /* Compute interpolation fractions */
                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                /* Compute moving image intensity using linear interpolation */
                /* Macro is slightly faster than function */
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving);

                /* Compute linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* Compute intensity difference */
                diff = m_val - f_img[fv];

                /* Compute spatial gradient using nearest neighbors */
                mvr = volume_index (moving->dim, mijk_r);
                dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
                dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
                dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
        
                if (parms->debug) {
                    fprintf (fp, "%u %u %u %g %g %g [%g]\n", 
                        (unsigned int) rijk[0], 
                        (unsigned int) rijk[1], 
                        (unsigned int) rijk[2], 
                        dc_dv[0], dc_dv[1], dc_dv[2],
                        diff);
                }

                score_acc += diff * diff;
                ssd->num_vox++;

            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_THRU_ROI_Z */

    if (parms->debug) {
        fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->smetric = score_acc / ssd->num_vox;
    for (int i = 0; i < bxf->num_coeff; i++) {
        ssd->grad[i] = 2 * ssd->grad[i] / ssd->num_vox;
    }

    ssd->time_smetric = plm_timer_report (timer);
    plm_timer_destroy (timer);
}

/* -----------------------------------------------------------------------
   FUNCTION: bspline_score_i_dcos_mse()

   Based on the "c" algorithm, but respects direction cosines.  
   This implementation computes both forward projection from fixed to 
   world, as well as backward projection from world to moving.
   ----------------------------------------------------------------------- */
void
bspline_score_i_mse (
    Bspline_optimize_data *bod
)
{
    Bspline_parms *parms = bod->parms;
    Bspline_state *bst = bod->bst;
    Bspline_xform *bxf = bod->bxf;

    Volume *fixed = parms->fixed;
    Volume *moving = parms->moving;
    Volume *moving_grad = parms->moving_grad;

    Bspline_score* ssd = &bst->ssd;
    plm_long fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];              /* Indices within moving image (vox) */
    float fxyz[3];              /* Position within fixed image (mm) */
    float mxyz[3];              /* Position within moving image (mm) */
    plm_long mijk_f[3], mvf;      /* Floor */
    plm_long mijk_r[3], mvr;      /* Round */
    plm_long p[3], pidx;          /* Region index of fixed voxel */
    plm_long q[3], qidx;          /* Offset index of fixed voxel */

    float dc_dv[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    float m_val;

    /* GCS: Oct 5, 2009.  We have determined that sequential accumulation
       of the score requires double precision.  However, reduction 
       accumulation does not. */
    double score_acc = 0.;

    static int it = 0;
    FILE* dc_dv_fp = 0;
    FILE* corr_fp = 0;

    if (parms->debug) {
        char buf[1024];
        sprintf (buf, "dc_dv_mse_%02d.txt", it);
        std::string fn = parms->debug_dir + "/" + buf;
        make_directory_recursive (fn.c_str());
        dc_dv_fp = fopen (fn.c_str(), "wb");

        sprintf (buf, "corr_mse_%02d.txt", it);
        fn = parms->debug_dir + "/" + buf;
        corr_fp = fopen (fn.c_str(), "wb");
        it ++;
    }

    Plm_timer* timer = plm_timer_create ();
    plm_timer_start (timer);

    ssd->num_vox = 0;
    ssd->smetric = 0.0f;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));

    /* GCS FIX: region of interest is not used */
    LOOP_Z (fijk, fxyz, fixed) {
        p[2] = REGION_INDEX_Z (fijk, bxf);
        q[2] = REGION_OFFSET_Z (fijk, bxf);
        LOOP_Y (fijk, fxyz, fixed) {
            p[1] = REGION_INDEX_Y (fijk, bxf);
            q[1] = REGION_OFFSET_Y (fijk, bxf);
            LOOP_X (fijk, fxyz, fixed) {
                p[0] = REGION_INDEX_X (fijk, bxf);
                q[0] = REGION_OFFSET_X (fijk, bxf);

                /* Get B-spline deformation vector */
                pidx = volume_index (bxf->rdims, p);
                qidx = volume_index (bxf->vox_per_rgn, q);
                bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                /* Compute moving image coordinate of fixed image voxel */
                mxyz[2] = fxyz[2] + dxyz[2] - moving->offset[2];
                mxyz[1] = fxyz[1] + dxyz[1] - moving->offset[1];
                mxyz[0] = fxyz[0] + dxyz[0] - moving->offset[0];
                mijk[2] = PROJECT_Z (mxyz, moving->proj);
                mijk[1] = PROJECT_Y (mxyz, moving->proj);
                mijk[0] = PROJECT_X (mxyz, moving->proj);

                if (parms->debug) {
                    fprintf (corr_fp, 
                        "%d %d %d %f %f %f\n",
                        (unsigned int) fijk[0], (unsigned int) fijk[1], 
                        (unsigned int) fijk[2], mijk[0], mijk[1], mijk[2]);
                }

                if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) continue;
                if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) continue;
                if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) continue;

                /* Compute interpolation fractions */
                li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

                /* Find linear index of "corner voxel" in moving image */
                mvf = volume_index (moving->dim, mijk_f);

                /* Compute moving image intensity using linear interpolation */
                /* Macro is slightly faster than function */
                LI_VALUE (m_val, 
                    li_1[0], li_2[0],
                    li_1[1], li_2[1],
                    li_1[2], li_2[2],
                    mvf, m_img, moving);

                /* Compute linear index of fixed image voxel */
                fv = volume_index (fixed->dim, fijk);

                /* Compute intensity difference */
                float diff = m_val - f_img[fv];

                /* Compute spatial gradient using nearest neighbors */
                mvr = volume_index (moving->dim, mijk_r);
                dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
                dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
                dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
                bspline_update_grad_b (&bst->ssd, bxf, pidx, qidx, dc_dv);
        
                if (parms->debug) {
                    fprintf (dc_dv_fp, 
                        "%u %u %u %g %g %g %g\n", 
                        (unsigned int) fijk[0], (unsigned int) fijk[1], 
                        (unsigned int) fijk[2], diff, dc_dv[0], 
                        dc_dv[1], dc_dv[2]);
                }

                score_acc += diff * diff;
                ssd->num_vox++;

            } /* LOOP_THRU_ROI_X */
        } /* LOOP_THRU_ROI_Y */
    } /* LOOP_THRU_ROI_Z */

    if (parms->debug) {
        fclose (dc_dv_fp);
        fclose (corr_fp);
    }

    /* Normalize score for MSE */
    ssd->smetric = score_acc / ssd->num_vox;
    for (int i = 0; i < bxf->num_coeff; i++) {
        ssd->grad[i] = 2 * ssd->grad[i] / ssd->num_vox;
    }

    ssd->time_smetric = plm_timer_report (timer);
    plm_timer_destroy (timer);
}
