/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
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

#include "bspline.h"
#include "bspline_mse_cpu_c.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_regularize.h"
#include "bspline_landmarks.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "delayload.h"
#include "interpolate.h"
#include "logfile.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_path.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_macros.h"
#include "xpm.h"

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
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    double score_tile;
    int num_vox;

    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;

    int idx_tile;
    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    Timer timer;
    double interval;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    int i;

    // Start timing the code
    plm_timer_start (&timer);

    // Zero out accumulators
    ssd->score = 0;
    num_vox = 0;
    score_tile = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

    // Serial across tiles
    for (idx_tile = 0; idx_tile < num_tiles; idx_tile++) {
	int rc;

	int crds_tile[3];
	int crds_local[3];
	int idx_local;

	float phys_fixed[3];
	int crds_fixed[3];
	int idx_fixed;

	float dxyz[3];

	float phys_moving[3];
	float crds_moving[3];
	int crds_moving_floor[3];
	int crds_moving_round[3];
	int idx_moving_floor;
	int idx_moving_round;

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
	COORDS_FROM_INDEX (crds_tile, idx_tile, bxf->rdims); 

	// Serial through voxels in tile
	for (crds_local[2] = 0; crds_local[2] < bxf->vox_per_rgn[2]; crds_local[2]++) {
	    for (crds_local[1] = 0; crds_local[1] < bxf->vox_per_rgn[1]; crds_local[1]++) {
		for (crds_local[0] = 0; crds_local[0] < bxf->vox_per_rgn[0]; crds_local[0]++) {
					
		    // Construct coordinates into fixed image volume
		    crds_fixed[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0] * crds_tile[0] + crds_local[0];
		    crds_fixed[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1] * crds_tile[1] + crds_local[1];
		    crds_fixed[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2] * crds_tile[2] + crds_local[2];

		    // Make sure we are inside the image volume
		    if (crds_fixed[0] >= bxf->roi_offset[0] + bxf->roi_dim[0])
			continue;
		    if (crds_fixed[1] >= bxf->roi_offset[1] + bxf->roi_dim[1])
			continue;
		    if (crds_fixed[2] >= bxf->roi_offset[2] + bxf->roi_dim[2])
			continue;

		    // Compute physical coordinates of fixed image voxel
		    phys_fixed[0] = bxf->img_origin[0] + bxf->img_spacing[0] * crds_fixed[0];
		    phys_fixed[1] = bxf->img_origin[1] + bxf->img_spacing[1] * crds_fixed[1];
		    phys_fixed[2] = bxf->img_origin[2] + bxf->img_spacing[2] * crds_fixed[2];
					
		    // Construct the local index within the tile
		    idx_local = INDEX_OF (crds_local, bxf->vox_per_rgn);

		    // Construct the image volume index
		    idx_fixed = INDEX_OF (crds_fixed, fixed->dim);

		    // Calc. deformation vector (dxyz) for voxel
		    bspline_interp_pix_b (dxyz, bxf, idx_tile, idx_local);

		    // Calc. moving image coordinate from the deformation vector
		    rc = bspline_find_correspondence (phys_moving,
			crds_moving,
			phys_fixed,
			dxyz,
			moving);

		    // Return code is 0 if voxel is pushed outside of moving image
		    if (!rc) continue;

		    // Compute linear interpolation fractions
		    li_clamp_3d (
			crds_moving,
			crds_moving_floor,
			crds_moving_round,
			li_1,
			li_2,
			moving);

		    // Find linear indices for moving image
		    idx_moving_floor = INDEX_OF (crds_moving_floor, moving->dim);
		    idx_moving_round = INDEX_OF (crds_moving_round, moving->dim);

		    // Calc. moving voxel intensity via linear interpolation
		    LI_VALUE (m_val, 
			li_1[0], li_2[0],
			li_1[1], li_2[1],
			li_1[2], li_2[2],
			idx_moving_floor,
			m_img, moving);

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
		    bspline_update_sets (sets_x, sets_y, sets_z,
			idx_local, dc_dv, bxf);
		}
	    }
	}
		
	// The tile is now condensed.  Now we will put it in the
	// proper slot within the control point bin that it belong to.
	bspline_sort_sets (cond_x, cond_y, cond_z,
	    sets_x, sets_y, sets_z,
	    idx_tile, bxf);


    }

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

    ssd->score = score_tile / num_vox;

    for (i = 0; i < bxf->num_coeff; i++) {
        ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
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
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    double score_tile;
    int num_vox;

    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;

    int idx_tile;
    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    Timer timer;
    double interval;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    int i;

    // Start timing the code
    plm_timer_start (&timer);

    // Zero out accumulators
    ssd->score = 0;
    num_vox = 0;
    score_tile = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

    // Parallel across tiles
#pragma omp parallel for reduction (+:num_vox,score_tile)
    for (idx_tile = 0; idx_tile < num_tiles; idx_tile++) {
	int rc;

	int crds_tile[3];
	int crds_local[3];
	int idx_local;

	float phys_fixed[3];
	int crds_fixed[3];
	int idx_fixed;

	float dxyz[3];

	float phys_moving[3];
	float crds_moving[3];
	int crds_moving_floor[3];
	int crds_moving_round[3];
	int idx_moving_floor;
	int idx_moving_round;

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
	COORDS_FROM_INDEX (crds_tile, idx_tile, bxf->rdims); 

	// Serial through voxels in tile
	for (crds_local[2] = 0; crds_local[2] < bxf->vox_per_rgn[2]; crds_local[2]++) {
	    for (crds_local[1] = 0; crds_local[1] < bxf->vox_per_rgn[1]; crds_local[1]++) {
		for (crds_local[0] = 0; crds_local[0] < bxf->vox_per_rgn[0]; crds_local[0]++) {
					
		    // Construct coordinates into fixed image volume
		    crds_fixed[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0] * crds_tile[0] + crds_local[0];
		    crds_fixed[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1] * crds_tile[1] + crds_local[1];
		    crds_fixed[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2] * crds_tile[2] + crds_local[2];

		    // Make sure we are inside the image volume
		    if (crds_fixed[0] >= bxf->roi_offset[0] + bxf->roi_dim[0])
			continue;
		    if (crds_fixed[1] >= bxf->roi_offset[1] + bxf->roi_dim[1])
			continue;
		    if (crds_fixed[2] >= bxf->roi_offset[2] + bxf->roi_dim[2])
			continue;

		    // Compute physical coordinates of fixed image voxel
		    phys_fixed[0] = bxf->img_origin[0] + bxf->img_spacing[0] * crds_fixed[0];
		    phys_fixed[1] = bxf->img_origin[1] + bxf->img_spacing[1] * crds_fixed[1];
		    phys_fixed[2] = bxf->img_origin[2] + bxf->img_spacing[2] * crds_fixed[2];
					
		    // Construct the local index within the tile
		    idx_local = INDEX_OF (crds_local, bxf->vox_per_rgn);

		    // Construct the image volume index
		    idx_fixed = INDEX_OF (crds_fixed, fixed->dim);

		    // Calc. deformation vector (dxyz) for voxel
		    bspline_interp_pix_b (dxyz, bxf, idx_tile, idx_local);

		    // Calc. moving image coordinate from the deformation vector
		    rc = bspline_find_correspondence (phys_moving,
			crds_moving,
			phys_fixed,
			dxyz,
			moving);

		    // Return code is 0 if voxel is pushed outside of 
		    // moving image
		    if (!rc) continue;

		    // Compute linear interpolation fractions
		    li_clamp_3d (
			crds_moving,
			crds_moving_floor,
			crds_moving_round,
			li_1,
			li_2,
			moving);

		    // Find linear indices for moving image
		    idx_moving_floor = INDEX_OF (crds_moving_floor, moving->dim);
		    idx_moving_round = INDEX_OF (crds_moving_round, moving->dim);

		    // Calc. moving voxel intensity via linear interpolation
		    LI_VALUE (m_val, 
			li_1[0], li_2[0],
			li_1[1], li_2[1],
			li_1[2], li_2[2],
			idx_moving_floor,
			m_img, moving);

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
            bspline_update_sets (sets_x, sets_y, sets_z,
                                 idx_local, dc_dv, bxf);
		}
	    }
	}

	// The tile is now condensed.  Now we will put it in the
	// proper slot within the control point bin that it belong to.
    bspline_sort_sets (cond_x, cond_y, cond_z,
                       sets_x, sets_y, sets_z,
                       idx_tile, bxf);
    }

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

    ssd->score = score_tile / num_vox;

    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}
