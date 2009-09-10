/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if defined (_WIN32)
#include <windows.h>
#endif
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_cuda.h"
#include "bspline_cuda_kernels.h"

// Define file-scope textures
texture<float, 1, cudaReadModeElementType> tex_fixed_image;
texture<float, 1, cudaReadModeElementType> tex_moving_image;
texture<float, 1, cudaReadModeElementType> tex_moving_grad;
texture<float, 1, cudaReadModeElementType> tex_coeff;
texture<int, 1, cudaReadModeElementType>   tex_c_lut;
texture<float, 1, cudaReadModeElementType> tex_q_lut;
texture<float, 1, cudaReadModeElementType> tex_score;

texture<float, 1> tex_dx;
texture<float, 1> tex_dy;
texture<float, 1> tex_dz;

texture<float, 1> tex_diff;
texture<float, 1> tex_mvr;

texture<float, 1> tex_dc_dv;
texture<float, 1> tex_dc_dv_x;
texture<float, 1> tex_dc_dv_y;
texture<float, 1> tex_dc_dv_z;
texture<float, 1> tex_grad;

// Define global variables.
float *gpu_fixed_image;  // The fixed image
float *gpu_moving_image; // The moving image
float *gpu_moving_grad;
int   *gpu_c_lut; // The c_lut indicating which control knots affect voxels within a region
float *gpu_q_lut; // The q_lut indicating the distance of a voxel to each of the 64 control knots
float *gpu_coeff; // The coefficient stream indicating the x, y, z coefficients of each control knot
float *gpu_dx; // Streams to store voxel displacement/gradient values in the X, Y, and Z directions 
float *gpu_dy; 
float *gpu_dz;
float *gpu_diff;
float *gpu_mvr;
float *gpu_dc_dv_x;
float *gpu_dc_dv_y;
float *gpu_dc_dv_z;
int   *gpu_valid_voxels;

float *gpu_dc_dv;
float *gpu_score;

size_t coeff_mem_size;
size_t dc_dv_mem_size;
size_t score_mem_size;

float *gpu_grad;
float *gpu_grad_temp;

/***********************************************************************
 * test_kernel
 *
 * A simple kernel used to ensure that CUDA is working correctly. 
 ***********************************************************************/
__global__ void
test_kernel
(
 int3 volume_dim,
 float *dx,
 float *dy,
 float *dz)
{
    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
	
    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    if (threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
    {
	dx[threadIdxInGrid] = (float)threadIdxInGrid;
	dy[threadIdxInGrid] = (float)threadIdxInGrid;
	dz[threadIdxInGrid] = (float)threadIdxInGrid;
    }
}

/***********************************************************************
 * bspline_cuda_score_g_mse_kernel1
 * 
 * This kernel calculates the values for the score and dc_dv streams.
 * It is similar to bspline_cuda_score_f_mse_kernel1, but it computes
 * the c_lut and q_lut values on the fly rather than referencing the
 * lookup tables.
 
 Updated by N. Kandasamy.
 Date: 07 July 2009.
 ***********************************************************************/
__global__ void
bspline_cuda_score_g_mse_kernel1 
(
 float  *dc_dv,
 float  *score,
 float  *coeff,
 float  *fixed_image,
 float  *moving_image,
 float  *moving_grad,
 int3   volume_dim,		// x, y, z dimensions of the volume in voxels
 float3 img_origin,		// Image origin (in mm)
 float3 img_spacing,     // Image spacing (in mm)
 float3 img_offset,		// Offset corresponding to the region of interest
 int3   roi_offset,	    // Position of first vox in ROI (in vox)
 int3   roi_dim,			// Dimension of ROI (in vox)
 int3   vox_per_rgn,	    // Knot spacing (in vox)
 float3 pix_spacing,		// Dimensions of a single voxel (in mm)
 int3   rdims,			// # of regions in (x,y,z)
 int3   cdims)
{
    extern __shared__ float sdata[]; 
	
    int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
    int3   p;				// Index of the tile within the volume (x,y,z)
    int3   q;				// Offset within the tile (measured in voxels)
    int    fv;				// Index of voxel in linear image array
    int    pidx;			// Index into c_lut
    int    qidx;			// Index into q_lut
    int    cidx;			// Index into the coefficient table

    float  P;				
    float3 N;				// Multiplier values
    float3 d;				// B-spline deformation vector
    float  diff;

    float3 distance_from_image_origin;
    float3 displacement_in_mm; 
    float3 displacement_in_vox;
    float3 displacement_in_vox_floor;
    float3 displacement_in_vox_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int    mvf;
    float  mvr;
    float  m_val;
    float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // Allocate memory for the spline coefficients evaluated at indices 0, 1, 2, and 3 in the 
    // X, Y, and Z directions
    float *A = &sdata[12*threadIdxInBlock + 0];
    float *B = &sdata[12*threadIdxInBlock + 4];
    float *C = &sdata[12*threadIdxInBlock + 8];
    float ii, jj, kk;
    float t1, t2, t3; 
    float one_over_six = 1.0/6.0;

    // If the voxel lies outside the volume, do nothing.
    if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
    {	
	// Calculate the x, y, and z coordinate of the voxel within the volume.
	coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
	coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
	coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
	// Calculate the x, y, and z offsets of the tile that contains this voxel.
	p.x = coord_in_volume.x / vox_per_rgn.x;
	p.y = coord_in_volume.y / vox_per_rgn.y;
	p.z = coord_in_volume.z / vox_per_rgn.z;
				
	// Calculate the x, y, and z offsets of the voxel within the tile.
	q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
	q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
	q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
	   coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
	   coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // Use the offset of the voxel within the region to compute the index into the c_lut.
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

	    // Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element = &dc_dv_element[3 * qidx];

	    // Compute the q_lut values that pertain to this offset.
	    ii = ((float)q.x) / vox_per_rgn.x;
	    t3 = ii*ii*ii;
	    t2 = ii*ii;
	    t1 = ii;
	    A[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    A[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    A[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    A[3] = one_over_six * (+ 1.0 * t3);

	    jj = ((float)q.y) / vox_per_rgn.y;
	    t3 = jj*jj*jj;
	    t2 = jj*jj;
	    t1 = jj;
	    B[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    B[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    B[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    B[3] = one_over_six * (+ 1.0 * t3);

	    kk = ((float)q.z) / vox_per_rgn.z;
	    t3 = kk*kk*kk;
	    t2 = kk*kk;
	    t1 = kk;
	    C[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    C[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    C[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    C[3] = one_over_six * (+ 1.0 * t3);

	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    // Compute the B-spline interpolant for the voxel
	    int3 t;
	    for(t.z = 0; t.z < 4; t.z++) {
		for(t.y = 0; t.y < 4; t.y++) {
		    for(t.x = 0; t.x < 4; t.x++) {

			// Calculate the index into the coefficients array.
			cidx = 3 * ((p.z + t.z) * cdims.x * cdims.y + (p.y + t.y) * cdims.x + (p.x + t.x));

			// Fetch the values for P, Ni, Nj, and Nk.
			P   = A[t.x] * B[t.y] * C[t.z];
			N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
			N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
			N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

			// Update the output (v) values.
			d.x += P * N.x;
			d.y += P * N.y;
			d.z += P * N.z;
		    }
		}
	    }

	    //-----------------------------------------------------------------
	    // Find correspondence in the moving image.
	    //-----------------------------------------------------------------

	    // Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
	    distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
	    distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
	    distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
	    // Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
	    displacement_in_mm.x = distance_from_image_origin.x + d.x;
	    displacement_in_mm.y = distance_from_image_origin.y + d.y;
	    displacement_in_mm.z = distance_from_image_origin.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
	    displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
	    displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

	    // Check if the displaced voxel lies outside the region of interest.
	    if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
		(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
		(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
		// Do nothing.
	    }
	    else {

		//-----------------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------------

		// Clamp and interpolate along the X axis.
		displacement_in_vox_floor.x = floor(displacement_in_vox.x);
		displacement_in_vox_round.x = round(displacement_in_vox.x);
		fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
		if(displacement_in_vox_floor.x < 0){
		    displacement_in_vox_floor.x = 0;
		    displacement_in_vox_round.x = 0;
		    fx2 = 0.0;
		}
		else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
		    displacement_in_vox_floor.x = volume_dim.x - 2;
		    displacement_in_vox_round.x = volume_dim.x - 1;
		    fx2 = 1.0;
		}
		fx1 = 1.0 - fx2;

		// Clamp and interpolate along the Y axis.
		displacement_in_vox_floor.y = floor(displacement_in_vox.y);
		displacement_in_vox_round.y = round(displacement_in_vox.y);
		fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
		if(displacement_in_vox_floor.y < 0){
		    displacement_in_vox_floor.y = 0;
		    displacement_in_vox_round.y = 0;
		    fy2 = 0.0;
		}
		else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
		    displacement_in_vox_floor.y = volume_dim.y - 2;
		    displacement_in_vox_round.y = volume_dim.y - 1;
		    fy2 = 1.0;
		}
		fy1 = 1.0 - fy2;
				
		// Clamp and intepolate along the Z axis.
		displacement_in_vox_floor.z = floor(displacement_in_vox.z);
		displacement_in_vox_round.z = round(displacement_in_vox.z);
		fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
		if(displacement_in_vox_floor.z < 0){
		    displacement_in_vox_floor.z = 0;
		    displacement_in_vox_round.z = 0;
		    fz2 = 0.0;
		}
		else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
		    displacement_in_vox_floor.z = volume_dim.z - 2;
		    displacement_in_vox_round.z = volume_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0 - fz2;
				
		//-----------------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------------

		mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
		m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);

		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		//-----------------------------------------------------------------
		// Compute intensity difference.
		//-----------------------------------------------------------------

		diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
		//-----------------------------------------------------------------
		// Accumulate the score.
		//-----------------------------------------------------------------

		score[threadIdxInGrid] = (diff * diff);

		//-----------------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------------
				
		// Compute spatial gradient using nearest neighbors.
		mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
		dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
		dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
		dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
	    }
	}
    }
}

/******************************************************
* This function computes the spline basis function at 
* index 0, 1, 2, or 3 for a voxel 
Author: Naga Kandasamy
Date: 07 July 2009
*******************************************************/

__device__ float
obtain_spline_basis_function (float one_over_six,
			      int t_idx, 
			      int vox_idx, 
			      int vox_per_rgn)
{
    float i = (float)vox_idx / vox_per_rgn;
    float C;
						
    switch(t_idx) {
    case 0:
	C = one_over_six * (- 1.0 * i*i*i + 3.0 * i*i - 3.0 * i + 1.0);
	break;
    case 1:
	C = one_over_six * (+ 3.0 * i*i*i - 6.0 * i*i            + 4.0);
	break;
    case 2:
	C = one_over_six * (- 3.0 * i*i*i + 3.0 * i*i + 3.0 * i + 1.0);
	break;
    case 3:
	C = one_over_six * (+ 1.0 * i*i*i);
	break;
    }
    return C;
}

/******************************************************************
* This function performs the gradient computation. It operates on 
* each control knot is parallel and each control knot accumulates 
* the influence of the 64 tiles on each control knot.

Updated by Naga Kandasamy
Date: 07 July 2009 
*******************************************************************/

__global__ void bspline_cuda_score_g_mse_kernel2 
(
 float *dc_dv,
 float *grad,
 int   num_threads,
 int3  rdims,
 int3  cdims,
 int3  vox_per_rgn)
{
    int3 knotLocation, tileOffset, tileLocation;
    int idx;
    int dc_dv_row;
    float A, B, C;
    int3 q;
    float one_over_six = 1.0/6.0;

    float3 result;
    result.x = 0.0;
    result.y = 0.0;
    result.z = 0.0;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
	
    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);
	
    // Next, calculate the index of the thread in its thread block, 
    // in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
	
    // Finally, calculate the index of the thread in the grid, based on 
    // the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // If the thread does not correspond to a control point, do nothing.
    if (threadIdxInGrid >= num_threads) {
	return;
    }

    // Determine the x, y, and z offset of the knot within the grid.
    knotLocation.x = threadIdxInGrid % cdims.x;
    knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
    knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

    // Subtract 1 from each of the knot indices to account for the 
    // differing origin between the knot grid and the tile grid.
    knotLocation.x -= 1;
    knotLocation.y -= 1;
    knotLocation.z -= 1;

    // Iterate through each of the 64 tiles that influence this 
    // control knot.
    for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
	for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
	    for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
		// Using the current x, y, and z offset from the control knot position,
		// calculate the index for one of the tiles that influence this knot.
		tileLocation.x = knotLocation.x + tileOffset.x;
		tileLocation.y = knotLocation.y + tileOffset.y;
		tileLocation.z = knotLocation.z + tileOffset.z;

		// Determine if the tile location is within the volume.
		if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
		   (tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
		   (tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

		    // Calculate linear index for tile.
		    idx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
		    // Calculate the offset into the dc_dv array corresponding to this tile.
		    dc_dv_row = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * idx;
		    int3 t;
		    t.x = abs(tileOffset.x - 1);
		    t.y = abs(tileOffset.y - 1);
		    t.z = abs(tileOffset.z - 1);
						
		    // For all the voxels in this tile, compute the influence on the control knot. We first compute the appropriate 
		    // spline paramterizationfor each voxel, relative to the control knot of interest
						
		    float pre_multiplier;
		    float multiplier_1, multiplier_2, multiplier_3, multiplier_4;
						
		    // Set this parameter to achieve the level of loop 
		    // unrolling desired; could be 1 or 4
		    // An unrolling factor of four appears to be the best 
		    // performer.  
		    int unrolling_factor = 4;
		    // The modified index is an integral multiple of 
		    // the unrolling factor
		    int modified_idx = (vox_per_rgn.x/unrolling_factor)*unrolling_factor; 
		    int lop_off = vox_per_rgn.x - modified_idx;
							
		    // Compute the spline parametization	
		    for(q.z = 0, idx = 0; q.z < vox_per_rgn.z; q.z++) {
			C = obtain_spline_basis_function(one_over_six, t.z, q.z, vox_per_rgn.z);	// Obtain the basis function along the Z direction
			for(q.y = 0; q.y < vox_per_rgn.y; q.y++) {
			    B = obtain_spline_basis_function(one_over_six, t.y, q.y, vox_per_rgn.y); // Obtain the basis function along the Y direction
			    pre_multiplier = B*C;
								
			    // The inner loop is unrolled multiple times as per a specified unrolling factor 
			    for(q.x = 0; q.x < modified_idx; q.x = q.x + unrolling_factor, idx = idx + unrolling_factor) {

				if(unrolling_factor == 1){ // No loop unrolling
				    A = obtain_spline_basis_function(one_over_six, t.x, q.x, vox_per_rgn.x); // Obtain the basis function for voxel in the X direction
				    multiplier_1 = A*pre_multiplier;
										
				    // Accumulate the results
				    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 0) * multiplier_1;
				    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 1) * multiplier_1;
				    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 2) * multiplier_1;	
				} // End if unrolling_factor = 1

				if(unrolling_factor == 4){ // The loop is unrolled four times 
				    A = obtain_spline_basis_function(one_over_six, t.x, q.x, vox_per_rgn.x); // Obtain the basis function for Voxel 1 in the X direction
				    multiplier_1 = A * pre_multiplier;
										
				    A = obtain_spline_basis_function(one_over_six, t.x, (q.x + 1), vox_per_rgn.x); // Obtain the basis function for Voxel 2 in the X direction
				    multiplier_2 = A * pre_multiplier;
										
				    A = obtain_spline_basis_function(one_over_six, t.x, (q.x + 2), vox_per_rgn.x); // Obtain the basis function for Voxel 3 in the X direction
				    multiplier_3 = A * pre_multiplier;
										
				    A = obtain_spline_basis_function(one_over_six, t.x, (q.x + 3), vox_per_rgn.x); // Obtain the basis function for Voxel 4 in the X direction
				    multiplier_4 = A * pre_multiplier;
										
				    // Accumulate the results
				    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 0) * multiplier_1;
				    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 1) * multiplier_1;
				    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 2) * multiplier_1;

				    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 1) + 0) * multiplier_2;
				    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 1) + 1) * multiplier_2;
				    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 1) + 2) * multiplier_2;
											
				    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 2) + 0) * multiplier_3;
				    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 2) + 1) * multiplier_3;
				    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 2) + 2) * multiplier_3;
											
				    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 3) + 0) * multiplier_4;
				    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 3) + 1) * multiplier_4;
				    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(idx + 3) + 2) * multiplier_4;
										
				} // End if unrolling_factor == 4
			    } // End for q.x loop
								
			    // Take care of any lop off voxels that the unrolled loop did not process
			    for(q.x = modified_idx; q.x < (modified_idx + lop_off); q.x++, idx++){
				A = obtain_spline_basis_function(one_over_six, t.x, q.x, vox_per_rgn.x); // Obtain the basis function for voxel in the X direction
				multiplier_1 = A * pre_multiplier;
										
				// Accumulate the results
				result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 0) * multiplier_1;
				result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 1) * multiplier_1;
				result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*idx + 2) * multiplier_1;
			    } // End of lop off loop
			} // End for q.y loop
		    } // End q.z loop
		}
	    }
	}
    }


    grad[3*threadIdxInGrid+0] = result.x;
    grad[3*threadIdxInGrid+1] = result.y;
    grad[3*threadIdxInGrid+2] = result.z;
}


/***********************************************************************
 * bspline_cuda_score_g_mse_kernel1_low_mem
 * 
 * This kernel calculates the values for the score and dc_dv streams.
 * It is similar to bspline_cuda_score_f_mse_kernel1, but it computes
 * the c_lut and q_lut values on the fly rather than referencing the
 * lookup tables.  Also, unlike bspline_cuda_score_g_mse_kernel1 above,
 * this version operates on only a portion of the volume at one time
 * in order to reduce the memory requirements on the GPU.
 
 Updated by Naga Kandasamy
 Date: 07 July 2009
 ***********************************************************************/
__global__ void
bspline_cuda_score_g_mse_kernel1_low_mem 
(
 float  *dc_dv,
 float  *score,			
 int    tile_index,		// Linear index of the starting tile
 int    num_tiles,       // Number of tiles to work on per kernel launch
 int3   volume_dim,		// x, y, z dimensions of the volume in voxels
 float3 img_origin,		// Image origin (in mm)
 float3 img_spacing,     // Image spacing (in mm)
 float3 img_offset,		// Offset corresponding to the region of interest
 int3   roi_offset,	    // Position of first vox in ROI (in vox)
 int3   roi_dim,			// Dimension of ROI (in vox)
 int3   vox_per_rgn,	    // Knot spacing (in vox)
 float3 pix_spacing,		// Dimensions of a single voxel (in mm)
 int3   rdims,			// # of regions in (x,y,z)
 int3   cdims)
{
    extern __shared__ float sdata[]; 
	
    int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
    int3   p;				// Offset of the tile within the volume
    int3   q;				// Offset within the tile (measured in voxels)
    int    fv;				// Index of voxel in linear image array
    int    pidx;			// Index into c_lut
    int    qidx;			// Index into q_lut
    int    cidx;			// Index into the coefficient table

    float  P;				
    float3 N;				// Multiplier values
    float3 d;				// B-spline deformation vector
    float  diff;

    float3 distance_from_image_origin;
    float3 displacement_in_mm; 
    float3 displacement_in_vox;
    float3 displacement_in_vox_floor;
    float3 displacement_in_vox_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int    mvf;
    float  mvr;
    float  m_val;
    float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    float *A = &sdata[12*threadIdxInBlock + 0];
    float *B = &sdata[12*threadIdxInBlock + 4];
    float *C = &sdata[12*threadIdxInBlock + 8];
    float ii, jj, kk;
    float t1, t2, t3; 
    float one_over_six = 1.0/6.0;

    // If the voxel lies outside this group of tiles, do nothing.
    if(threadIdxInGrid < (num_tiles * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
    {	
	// Update the tile index to store the index of the tile corresponding to this thread.
	tile_index += (threadIdxInGrid / (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z));

	// Determine the corresponding x, y, and z coordinates of the tile.
	p.x = tile_index % rdims.x;
	p.y = ((tile_index - p.x) / rdims.x) % rdims.y;
	p.z = ((((tile_index - p.x) / rdims.x) - p.y) / rdims.y) % rdims.z;

	// Calculate the x, y and z offsets of the voxel within the tile.
	q.x = threadIdxInGrid % vox_per_rgn.x;
	q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
	q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

	// Calculate the x, y and z offsets of the voxel within the volume.
	coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
	coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
	coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
	   coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
	   coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // Use the offset of the voxel within the region to compute the index into the c_lut.
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

	    // Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element = &dc_dv_element[3 * qidx];
			
	    // Compute the q_lut values that pertain to this offset.
	    ii = ((float)q.x) / vox_per_rgn.x;
	    t3 = ii*ii*ii;
	    t2 = ii*ii;
	    t1 = ii;
	    A[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    A[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    A[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    A[3] = one_over_six * (+ 1.0 * t3);

	    jj = ((float)q.y) / vox_per_rgn.y;
	    t3 = jj*jj*jj;
	    t2 = jj*jj;
	    t1 = jj;
	    B[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    B[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    B[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    B[3] = one_over_six * (+ 1.0 * t3);

	    kk = ((float)q.z) / vox_per_rgn.z;
	    t3 = kk*kk*kk;
	    t2 = kk*kk;
	    t1 = kk;
	    C[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    C[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    C[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    C[3] = one_over_six * (+ 1.0 * t3);

	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    int3 t;
	    for(t.z = 0; t.z < 4; t.z++) {
		for(t.y = 0; t.y < 4; t.y++) {
		    for(t.x = 0; t.x < 4; t.x++) {

			// Calculate the index into the coefficients array.
			cidx = 3 * ((p.z + t.z) * cdims.x * cdims.y + (p.y + t.y) * cdims.x + (p.x + t.x));

			// Fetch the values for P, Ni, Nj, and Nk.
			P   = A[t.x] * B[t.y] * C[t.z];
			N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
			N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
			N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

			// Update the output (v) values.
			d.x += P * N.x;
			d.y += P * N.y;
			d.z += P * N.z;
		    }
		}
	    }

	    //-----------------------------------------------------------------
	    // Find correspondence in the moving image.
	    //-----------------------------------------------------------------

	    // Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
	    distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
	    distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
	    distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
	    // Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
	    displacement_in_mm.x = distance_from_image_origin.x + d.x;
	    displacement_in_mm.y = distance_from_image_origin.y + d.y;
	    displacement_in_mm.z = distance_from_image_origin.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
	    displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
	    displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

	    // Check if the displaced voxel lies outside the region of interest.
	    if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
		(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
		(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
		// Do nothing.
	    }
	    else {

		//-----------------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------------

		// Clamp and interpolate along the X axis.
		displacement_in_vox_floor.x = floor(displacement_in_vox.x);
		displacement_in_vox_round.x = round(displacement_in_vox.x);
		fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
		if(displacement_in_vox_floor.x < 0){
		    displacement_in_vox_floor.x = 0;
		    displacement_in_vox_round.x = 0;
		    fx2 = 0.0;
		}
		else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
		    displacement_in_vox_floor.x = volume_dim.x - 2;
		    displacement_in_vox_round.x = volume_dim.x - 1;
		    fx2 = 1.0;
		}
		fx1 = 1.0 - fx2;

		// Clamp and interpolate along the Y axis.
		displacement_in_vox_floor.y = floor(displacement_in_vox.y);
		displacement_in_vox_round.y = round(displacement_in_vox.y);
		fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
		if(displacement_in_vox_floor.y < 0){
		    displacement_in_vox_floor.y = 0;
		    displacement_in_vox_round.y = 0;
		    fy2 = 0.0;
		}
		else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
		    displacement_in_vox_floor.y = volume_dim.y - 2;
		    displacement_in_vox_round.y = volume_dim.y - 1;
		    fy2 = 1.0;
		}
		fy1 = 1.0 - fy2;
				
		// Clamp and intepolate along the Z axis.
		displacement_in_vox_floor.z = floor(displacement_in_vox.z);
		displacement_in_vox_round.z = round(displacement_in_vox.z);
		fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
		if(displacement_in_vox_floor.z < 0){
		    displacement_in_vox_floor.z = 0;
		    displacement_in_vox_round.z = 0;
		    fz2 = 0.0;
		}
		else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
		    displacement_in_vox_floor.z = volume_dim.z - 2;
		    displacement_in_vox_round.z = volume_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0 - fz2;
				
		//-----------------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------------

		mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
		m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		//-----------------------------------------------------------------
		// Compute intensity difference.
		//-----------------------------------------------------------------

		diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
		//-----------------------------------------------------------------
		// Accumulate the score.
		//-----------------------------------------------------------------

		score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

		//-----------------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------------
				
		// Compute spatial gradient using nearest neighbors.
		mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
		dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
		dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
		dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			
	    }
	}
    }
}
							

/***********************************************************************
 * bspline_cuda_score_f_mse_compute_score
 *
 * This kernel computes only the diff, score, and mvr values.  It stores
 * each in streams to be used by bspline_cuda_score_f_compute_dc_dv. 
 * Separating the score and dc_dv calculations into two separate kernels
 * makes it possible to ensure writes to memory are coalesced.
 ***********************************************************************/
__global__ void 
bspline_cuda_score_f_mse_compute_score 
(
 float  *dc_dv,
 float  *score,
 float  *diffs,
 float  *mvrs,
 int3   volume_dim,		// x, y, z dimensions of the volume in voxels
 float3 img_origin,		// Image origin (in mm)
 float3 img_spacing,     // Image spacing (in mm)
 float3 img_offset,		// Offset corresponding to the region of interest
 int3   roi_offset,	    // Position of first vox in ROI (in vox)
 int3   roi_dim,			// Dimension of ROI (in vox)
 int3   vox_per_rgn,	    // Knot spacing (in vox)
 float3 pix_spacing,		// Dimensions of a single voxel (in mm)
 int3   rdims)			// # of regions in (x,y,z)
{
    int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
    int3   p;				// Index of the tile within the volume (x,y,z)
    int3   q;				// Offset within the tile (measured in voxels)
    int    fv;				// Index of voxel in linear image array
    int    pidx;			// Index into c_lut
    int    qidx;			// Index into q_lut
    int    cidx;			// Index into the coefficient table

    float  P;				
    float3 N;				// Multiplier values
    float3 d;				// B-spline deformation vector
    float  diff;

    float3 distance_from_image_origin;
    float3 displacement_in_mm; 
    float3 displacement_in_vox;
    float3 displacement_in_vox_floor;
    float3 displacement_in_vox_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int    mvf;
    float  mvr;
    float  m_val;
    float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // If the voxel lies outside the volume, do nothing.
    if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
    {	
	// Calculate the x, y, and z coordinate of the voxel within the volume.
	coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
	coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
	coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
	// Calculate the x, y, and z offsets of the tile that contains this voxel.
	p.x = coord_in_volume.x / vox_per_rgn.x;
	p.y = coord_in_volume.y / vox_per_rgn.y;
	p.z = coord_in_volume.z / vox_per_rgn.z;
				
	// Calculate the x, y, and z offsets of the voxel within the tile.
	q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
	q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
	q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
	   coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
	   coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // Use the offset of the voxel within the region to compute the index into the c_lut.
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];
	    pidx = pidx * 64;

	    // Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element = &dc_dv_element[3 * qidx];
	    qidx = qidx * 64;
			
	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    for(int k = 0; k < 64; k++)
	    {
		// Calculate the index into the coefficients array.
		cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
		// Fetch the values for P, Ni, Nj, and Nk.
		P   = tex1Dfetch(tex_q_lut, qidx + k); 
		N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
		N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
		N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

		// Update the output (v) values.
		d.x += P * N.x;
		d.y += P * N.y;
		d.z += P * N.z;
	    }
			
	    //-----------------------------------------------------------------
	    // Find correspondence in the moving image.
	    //-----------------------------------------------------------------

	    // Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
	    distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
	    distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
	    distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
	    // Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
	    displacement_in_mm.x = distance_from_image_origin.x + d.x;
	    displacement_in_mm.y = distance_from_image_origin.y + d.y;
	    displacement_in_mm.z = distance_from_image_origin.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
	    displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
	    displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

	    // Check if the displaced voxel lies outside the region of interest.
	    if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
		(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
		(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
		// Do nothing.
	    }
	    else {

		//-----------------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------------

		// Clamp and interpolate along the X axis.
		displacement_in_vox_floor.x = floor(displacement_in_vox.x);
		displacement_in_vox_round.x = round(displacement_in_vox.x);
		fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
		if(displacement_in_vox_floor.x < 0){
		    displacement_in_vox_floor.x = 0;
		    displacement_in_vox_round.x = 0;
		    fx2 = 0.0;
		}
		else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
		    displacement_in_vox_floor.x = volume_dim.x - 2;
		    displacement_in_vox_round.x = volume_dim.x - 1;
		    fx2 = 1.0;
		}
		fx1 = 1.0 - fx2;

		// Clamp and interpolate along the Y axis.
		displacement_in_vox_floor.y = floor(displacement_in_vox.y);
		displacement_in_vox_round.y = round(displacement_in_vox.y);
		fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
		if(displacement_in_vox_floor.y < 0){
		    displacement_in_vox_floor.y = 0;
		    displacement_in_vox_round.y = 0;
		    fy2 = 0.0;
		}
		else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
		    displacement_in_vox_floor.y = volume_dim.y - 2;
		    displacement_in_vox_round.y = volume_dim.y - 1;
		    fy2 = 1.0;
		}
		fy1 = 1.0 - fy2;
				
		// Clamp and intepolate along the Z axis.
		displacement_in_vox_floor.z = floor(displacement_in_vox.z);
		displacement_in_vox_round.z = round(displacement_in_vox.z);
		fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
		if(displacement_in_vox_floor.z < 0){
		    displacement_in_vox_floor.z = 0;
		    displacement_in_vox_round.z = 0;
		    fz2 = 0.0;
		}
		else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
		    displacement_in_vox_floor.z = volume_dim.z - 2;
		    displacement_in_vox_round.z = volume_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0 - fz2;
				
		//-----------------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------------

		mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
		m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		//-----------------------------------------------------------------
		// Compute intensity difference.
		//-----------------------------------------------------------------

		diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
		//-----------------------------------------------------------------
		// Accumulate the score.
		//-----------------------------------------------------------------

		score[threadIdxInGrid] = (diff * diff);

		diffs[threadIdxInGrid] = diff;

		//-----------------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------------
				
		// Compute spatial gradient using nearest neighbors.
		mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
		mvrs[threadIdxInGrid] = (float)mvr;	

		/*
		  dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
		  dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
		  dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
		*/
	    }
	}
    }
}

/***********************************************************************
 * bspline_cuda_score_f_mse_compute_dc_dv
 *
 * This kernel computes only the dc_dv values.
 * Separating the score and dc_dv calculations into two separate kernels
 * makes it possible to ensure writes to memory are coalesced.
 ***********************************************************************/
__global__ void bspline_cuda_score_f_compute_dc_dv(
	float *dc_dv,	
	int3  volume_dim,		// x, y, z dimensions of the volume in voxels
	int3  vox_per_rgn,	    // Knot spacing (in vox)
	int3  roi_offset,	    // Position of first vox in ROI (in vox)
	int3  roi_dim,			// Dimension of ROI (in vox)
	int3  rdims)			// # of regions in (x,y,z)
{	
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	int voxelIdx = threadIdxInGrid / 3;
	int xyzOffset = threadIdxInGrid - (3 * voxelIdx);

	int3  coord_in_volume;	// Coordinate of the voxel in the volume (x,y,z)
	int3  p;				// Index of the tile within the volume (x,y,z)
	int3  q;				// Offset within the tile (measured in voxels)
	int   pidx;				// Index into c_lut
	int   qidx;				// Index into q_lut
	float diff;
	float mvr;
	float *dc_dv_element;

	// If the voxel lies outside the volume, do nothing.
	if(voxelIdx < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the x, y, and z coordinate of the voxel within the volume.
		coord_in_volume.z = voxelIdx / (volume_dim.x * volume_dim.y);
		coord_in_volume.y = (voxelIdx - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		coord_in_volume.x = voxelIdx - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
		// Calculate the x, y, and z offsets of the tile that contains this voxel.
		p.x = coord_in_volume.x / vox_per_rgn.x;
		p.y = coord_in_volume.y / vox_per_rgn.y;
		p.z = coord_in_volume.z / vox_per_rgn.z;
				
		// Calculate the x, y, and z offsets of the voxel within the tile.
		q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
		q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
		q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

		// If the voxel lies outside of the region of interest, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_element = &dc_dv_element[3 * qidx];

			diff = tex1Dfetch(tex_diff, voxelIdx);
			mvr = tex1Dfetch(tex_mvr, voxelIdx);

			dc_dv_element[xyzOffset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + xyzOffset);
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel1_v2
 * 
 * This kernel fills the score and dc_dv streams.  It operates on the
 * entire volume at one time rather than performing the calculations
 * tile by tile.  An equivalent version that operates tile by tile is
 * given below (bspline_cuda_score_f_mse_kernel1_low_mem).  The score
 * stream should have the same number of elements are there are voxels
 * in the volume.
 ***********************************************************************/
__global__ void bspline_cuda_score_f_mse_kernel1_v2 (
	float  *dc_dv_x,
	float  *dc_dv_y,
	float  *dc_dv_z,
	float  *score,			
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	int3   rdims)			// # of regions in (x,y,z)
{
	int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
	int3   p;				// Index of the tile within the volume (x,y,z)
	int3   q;				// Offset within the tile (measured in voxels)
	int    fv;				// Index of voxel in linear image array
	int    pidx;			// Index into c_lut
	int    qidx;			// Index into q_lut
	int    cidx;			// Index into the coefficient table

	float  P;				
	float3 N;				// Multiplier values
	float3 d;				// B-spline deformation vector
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
	int dc_dv_offset;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the x, y, and z coordinate of the voxel within the volume.
		coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
		// Calculate the x, y, and z offsets of the tile that contains this voxel.
		p.x = coord_in_volume.x / vox_per_rgn.x;
		p.y = coord_in_volume.y / vox_per_rgn.y;
		p.z = coord_in_volume.z / vox_per_rgn.z;
				
		// Calculate the x, y, and z offsets of the voxel within the tile.
		q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
		q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
		q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

		// If the voxel lies outside of the region of interest, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_offset = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_offset += qidx;
			qidx = qidx * 64;
			
			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;

				dc_dv_x[dc_dv_offset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv_y[dc_dv_offset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv_z[dc_dv_offset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);

			}
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel1
 * 
 * This kernel fills the score and dc_dv streams.  It operates on the
 * entire volume at one time rather than performing the calculations
 * tile by tile.  An equivalent version that operates tile by tile is
 * given below (bspline_cuda_score_f_mse_kernel1_low_mem).  The score
 * stream should have the same number of elements are there are voxels
 * in the volume.
 
 Updated by Naga Kandasamy
 Date: 07 July 2009
 ***********************************************************************/
__global__ void bspline_cuda_score_f_mse_kernel1 (
	float  *dc_dv,
	float  *score,	
	int    *gpu_c_lut,
	float  *gpu_q_lut,
	float  *coeff,
	float  *fixed_image,
	float  *moving_image,
	float  *moving_grad,
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	int3   rdims)			// # of regions in (x,y,z)
{
	int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
	int3   p;				// Index of the tile within the volume (x,y,z)
	int3   q;				// Offset within the tile (measured in voxels)
	int    fv;				// Index of voxel in linear image array
	int    pidx;			// Index into c_lut
	int    qidx;			// Index into q_lut
	int    cidx;			// Index into the coefficient table

	float  P;				
	float3 N;				// Multiplier values
	float3 d;				// B-spline deformation vector
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
	float* dc_dv_element;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{	
		// Calculate the x, y, and z coordinate of the voxel within the volume.
		coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
		// Calculate the x, y, and z offsets of the tile that contains this voxel.
		p.x = coord_in_volume.x / vox_per_rgn.x;
		p.y = coord_in_volume.y / vox_per_rgn.y;
		p.z = coord_in_volume.z / vox_per_rgn.z;
				
		// Calculate the x, y, and z offsets of the voxel within the tile.
		q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
		q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
		q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

		// If the voxel lies outside of the region of interest, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_element = &dc_dv_element[3 * qidx]; // dc_dv_element+(3*qidx);

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				// cidx = 3 * gpu_c_lut[pidx + k];

				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				// P   = gpu_q_lut[qidx + k];
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// N.x = coeff[cidx+0];  // x-value
				// N.y = coeff[cidx+1];  // y-value
				// N.z = coeff[cidx+2];  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}

			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				// diff = fixed_image[fv] - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
				dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
				// dc_dv_element[0] = diff * moving_grad[3 * (int)mvr + 0];
				// dc_dv_element[1] = diff * moving_grad[3 * (int)mvr + 1];
				// dc_dv_element[2] = diff * moving_grad[3 * (int)mvr + 2];	
			}
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel1_low_mem
 * 
 * This kernel fills the score and dc_dv streams.  It performs its
 * calculations on a tile by tile basis, and therefore requires the
 * tile index (x, y, and z) as an input.  It uses less memory than 
 * bspline_cuda_score_f_mse_kernel1, but the performance is worse.
 * The score stream need only have the same number of elements as there
 * are voxels in a tile.
 ***********************************************************************/
__global__ void bspline_cuda_score_f_mse_kernel1_low_mem (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
	float* dc_dv_element;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInGrid < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			dc_dv_element = &dc_dv_element[3 * threadIdxInGrid];
			qidx = threadIdxInGrid * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
								
				dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

__global__ void bspline_cuda_score_f_mse_kernel2_v2 (
						     float *grad,
						     int   num_threads,
						     int3  rdims,
						     int3  cdims,
						     int3  vox_per_rgn)
{
    // Shared memory is allocated on a per block basis.  Therefore, only allocate 
    // (sizeof(data) * blocksize) memory when calling the kernel.
    extern __shared__ float sdata[]; 

    int3 knotLocation;
    int3 tileOffset;
    int3 tileLocation;
    int pidx;
    int qidx;
    int dc_dv_row;
    int m;	
    float multiplier;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    int totalVoxPerRgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

    float *temps = &sdata[15*threadIdxInBlock];
    temps[12] = 0.0;
    temps[13] = 0.0;
    temps[14] = 0.0;

    // If the thread does not correspond to a control point, do nothing.
    if(threadIdxInGrid < num_threads) {	

	// Determine the x, y, and z offset of the knot within the grid.
	knotLocation.x = threadIdxInGrid % cdims.x;
	knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
	knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

	// Subtract 1 from each of the knot indices to account for the differing origin
	// between the knot grid and the tile grid.
	knotLocation.x -= 1;
	knotLocation.y -= 1;
	knotLocation.z -= 1;

	// Iterate through each of the 64 tiles that influence this control knot.
	for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
	    for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
		for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
		    // Using the current x, y, and z offset from the control knot position,
		    // calculate the index for one of the tiles that influence this knot.
		    tileLocation.x = knotLocation.x + tileOffset.x;
		    tileLocation.y = knotLocation.y + tileOffset.y;
		    tileLocation.z = knotLocation.z + tileOffset.z;

		    // Determine if the tile location is within the volume.
		    if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
		       (tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
		       (tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

			// Calculate linear index for tile.
			pidx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
			// Calculate the offset into the dc_dv array corresponding to this tile.
			dc_dv_row = totalVoxPerRgn * pidx;

			// Update pidx to index into the c_lut.
			pidx = 64 * pidx;

			// Find the coefficient index in the c_lut row in order to determine
			// the linear index of the control point with respect to the current tile.
			for(m = 0; m < 64; m++) {
			    if(tex1Dfetch(tex_c_lut, pidx + m) == threadIdxInGrid) {
				break;
			    }
			}

			/*
			  for(qidx = 0; qidx < totalVoxPerRgn; qidx += 1) {
			  multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			  temps[12]  += tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx) * multiplier;
			  temps[13]  += tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx) * multiplier;
			  temps[14]  += tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx) * multiplier;
			  }
			*/

			for(qidx = 0; qidx < totalVoxPerRgn - 4; qidx = qidx + 4) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			    temps[6]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 2) * multiplier;
			    temps[7]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 2) * multiplier;
			    temps[8]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			    temps[9]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 3) * multiplier;
			    temps[10]  = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 3) * multiplier;
			    temps[11]  = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 3) * multiplier;

			    temps[12]  += temps[0] + temps[3] + temps[6] + temps[9];
			    temps[13]  += temps[1] + temps[4] + temps[7] + temps[10];
			    temps[14]  += temps[2] + temps[5] + temps[8] + temps[11];
			}
						
			if(qidx+3 < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			    temps[6]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 2) * multiplier;
			    temps[7]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 2) * multiplier;
			    temps[8]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			    temps[9]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 3) * multiplier;
			    temps[10]  = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 3) * multiplier;
			    temps[11]  = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 3) * multiplier;

			    temps[12]  += temps[0] + temps[3] + temps[6] + temps[9];
			    temps[13]  += temps[1] + temps[4] + temps[7] + temps[10];
			    temps[14]  += temps[2] + temps[5] + temps[8] + temps[11];
			}

			else if(qidx+2 < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			    temps[6]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 2) * multiplier;
			    temps[7]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 2) * multiplier;
			    temps[8]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 2) * multiplier;

			    temps[12]  += temps[0] + temps[3] + temps[6];
			    temps[13]  += temps[1] + temps[4] + temps[7];
			    temps[14]  += temps[2] + temps[5] + temps[8];
			}

			else if(qidx+1 < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    temps[12]  += temps[0] + temps[3];
			    temps[13]  += temps[1] + temps[4];
			    temps[14]  += temps[2] + temps[5];
			}

			else if(qidx < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[12]  += tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[13]  += tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[14]  += tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;
			}
		    }
		}
	    }
	}

	grad[3*threadIdxInGrid+0] = temps[12];
	grad[3*threadIdxInGrid+1] = temps[13];
	grad[3*threadIdxInGrid+2] = temps[14];

    }
}

__global__ void bspline_cuda_score_f_mse_kernel2 (
	float *dc_dv,
	float *grad,
	int   num_threads,
	int3  rdims,
	int3  cdims,
	int3  vox_per_rgn)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	int3 knotLocation;
	int3 tileOffset;
	int3 tileLocation;
	int pidx;
	int qidx;
	int dc_dv_row;
	int m;	
	float multiplier;

	/*
	float3 temp0, temp1, temp2, temp3;
	float3 result;
	result.x = 0.0;
	result.y = 0.0;
	result.z = 0.0;
	*/

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	int totalVoxPerRgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

	float *temps = &sdata[15*threadIdxInBlock];
	temps[12] = 0.0;
	temps[13] = 0.0;
	temps[14] = 0.0;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads) {	

		// Determine the x, y, and z offset of the knot within the grid.
		knotLocation.x = threadIdxInGrid % cdims.x;
		knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
		knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

		// Subtract 1 from each of the knot indices to account for the differing origin
		// between the knot grid and the tile grid.
		knotLocation.x -= 1;
		knotLocation.y -= 1;
		knotLocation.z -= 1;

		// Iterate through each of the 64 tiles that influence this control knot.
		for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
			for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
				for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
					// Using the current x, y, and z offset from the control knot position,
					// calculate the index for one of the tiles that influence this knot.
					tileLocation.x = knotLocation.x + tileOffset.x;
					tileLocation.y = knotLocation.y + tileOffset.y;
					tileLocation.z = knotLocation.z + tileOffset.z;

					// Determine if the tile location is within the volume.
					if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
						(tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
						(tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

						// Calculate linear index for tile.
						pidx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
						// Calculate the offset into the dc_dv array corresponding to this tile.
						dc_dv_row = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx;

						// Update pidx to index into the c_lut.
						pidx = 64 * pidx;

						// Find the coefficient index in the c_lut row in order to determine
						// the linear index of the control point with respect to the current tile.
						for(m = 0; m < 64; m++) {
							if(tex1Dfetch(tex_c_lut, pidx + m) == threadIdxInGrid) {
								break;
							}
						}									

						/*
						for(qidx = 0; qidx < totalVoxPerRgn - 4; qidx = qidx + 4) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temp2.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temp2.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temp2.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temp3.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temp3.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temp3.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							result.x += temp0.x + temp1.x + temp2.x + temp3.x;
							result.y += temp0.y + temp1.y + temp2.y + temp3.y;
							result.z += temp0.z + temp1.z + temp2.z + temp3.z;
						}
						
						if(qidx+3 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temp2.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temp2.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temp2.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temp3.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temp3.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temp3.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							result.x += temp0.x + temp1.x + temp2.x + temp3.x;
							result.y += temp0.y + temp1.y + temp2.y + temp3.y;
							result.z += temp0.z + temp1.z + temp2.z + temp3.z;
						}

						else if(qidx+2 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temp2.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temp2.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temp2.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							result.x += temp0.x + temp1.x + temp2.x;
							result.y += temp0.y + temp1.y + temp2.y;
							result.z += temp0.z + temp1.z + temp2.z;
						}

						else if(qidx+1 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							result.x += temp0.x + temp1.x;
							result.y += temp0.y + temp1.y;
							result.z += temp0.z + temp1.z;
						}

						else if(qidx < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;
						}
						*/

						for(qidx = 0; qidx < totalVoxPerRgn - 4; qidx = qidx + 4) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temps[6]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temps[7]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temps[8]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temps[9]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temps[10] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temps[11] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							temps[12] += temps[0] + temps[3] + temps[6] + temps[9];
							temps[13] += temps[1] + temps[4] + temps[7] + temps[10];
							temps[14] += temps[2] + temps[5] + temps[8] + temps[11];
						}
						
						if(qidx+3 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temps[6]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temps[7]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temps[8]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temps[9]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temps[10] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temps[11] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							temps[12] += temps[0] + temps[3] + temps[6] + temps[9];
							temps[13] += temps[1] + temps[4] + temps[7] + temps[10];
							temps[14] += temps[2] + temps[5] + temps[8] + temps[11];
						}

						else if(qidx+2 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temps[6]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temps[7]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temps[8]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							temps[12] += temps[0] + temps[3] + temps[6];
							temps[13] += temps[1] + temps[4] + temps[7];
							temps[14] += temps[2] + temps[5] + temps[8];
						}

						else if(qidx+1 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							temps[12] += temps[0] + temps[3];
							temps[13] += temps[1] + temps[4];
							temps[14] += temps[2] + temps[5];
						}

						else if(qidx < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[12] += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[13] += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[14] += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;
						}
					}
				}
			}
		}

		/*
		grad[3*threadIdxInGrid+0] = result.x;
		grad[3*threadIdxInGrid+1] = result.y;
		grad[3*threadIdxInGrid+2] = result.z;
		*/

		grad[3*threadIdxInGrid+0] = temps[12];
		grad[3*threadIdxInGrid+1] = temps[13];
		grad[3*threadIdxInGrid+2] = temps[14];

	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel2
 *
 * This kernel fills up the gradient stream.  Each thread represents one
 * control knot, and therefore one element in the gradient stream.  The
 * kernel determines which tiles influence the given control knot, 
 * iterates through the voxels of each of those tiles, and accumulates
 * the total influence.  It then saves the result to the gradient stream.
 * This implementation offers much better performance than any of the
 * previous versions, which calculate the gradient values on a tile by
 * tile basis.
 ***********************************************************************/
__global__ void 
bspline_cuda_score_f_mse_kernel2_nk 
(
 float *dc_dv,
 float *grad,
 int   num_threads,
 int3  rdims,
 int3  cdims,
 int3  vox_per_rgn)
{
    int3 knotLocation;
    int3 tileOffset;
    int3 tileLocation;
    int pidx;
    int qidx;
    int dc_dv_row;
    int m;	
    float multiplier;

    float3 result;
    result.x = 0.0; 
    result.y = 0.0;
    result.z = 0.0;
	
    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    //int totalVoxPerRgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

    // If the thread does not correspond to a control point, do nothing.
    if(threadIdxInGrid < num_threads) {	

	// Determine the x, y, and z offset of the knot within the grid.
	knotLocation.x = threadIdxInGrid % cdims.x;
	knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
	knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

	// Subtract 1 from each of the knot indices to account for the differing origin
	// between the knot grid and the tile grid.
	knotLocation.x -= 1;
	knotLocation.y -= 1;
	knotLocation.z -= 1;

	// Iterate through each of the 64 tiles that influence this control knot.
	for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
	    for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
		for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
		    // Using the current x, y, and z offset from the control knot position,
		    // calculate the index for one of the tiles that influence this knot.
		    tileLocation.x = knotLocation.x + tileOffset.x;
		    tileLocation.y = knotLocation.y + tileOffset.y;
		    tileLocation.z = knotLocation.z + tileOffset.z;

		    // Determine if the tile location is within the volume.
		    if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
		       (tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
		       (tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

			// Calculate linear index for tile.
			pidx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
			// Calculate the offset into the dc_dv array corresponding to this tile.
			dc_dv_row = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx;

			// Update pidx to index into the c_lut.
			pidx = 64 * pidx;

			// Find the coefficient index in the c_lut row in order to determine
			// the linear index of the control point with respect to the current tile.
			for(m = 0; m < 64; m++) {
			    if(tex1Dfetch(tex_c_lut, pidx + m) == threadIdxInGrid) 
				break;
			}									
						
			// Accumulate the influence of each voxel in the current tile
						
			// To improve performance, we unroll the loop to operate 
			// on multiple voxels per iteration. An unrolling factor of four appears to be the best performer
						
			int unrolling_factor = 4; // Set this parameter to achieve the level of loop unrolling desired; could be 1 or 4
			int total_vox_per_rgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
			int modified_idx = (total_vox_per_rgn/unrolling_factor)*unrolling_factor; // The modified index is an integral multiple of the unrolling factor
			int lop_off = total_vox_per_rgn - modified_idx;
						
			for(qidx = 0; qidx < modified_idx; qidx = qidx + unrolling_factor) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*qidx + m); // Voxel 1
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m); // Voxel 2
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m); // Voxel 3
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m); // Voxel 4
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;
			}
						
			// Take care of any lop off voxels
			for(qidx = modified_idx; qidx < (modified_idx + lop_off); qidx++){
			    multiplier = tex1Dfetch(tex_q_lut, 64*qidx + m);
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 2) * multiplier;
			} 
		    } // if tile location is within the volume
		} // for each tile
	    } // 
	}
	grad[3*threadIdxInGrid+0] = result.x;
	grad[3*threadIdxInGrid+1] = result.y;
	grad[3*threadIdxInGrid+2] = result.z;
		
    }
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel1a
 *
 * This kernel fills the score stream.  It operates on the entire volume
 * at one time, and therefore the number of elements in the score stream
 * must be equal to the number of voxels in the volume.  The dc_dv
 * computations are contained in a separate kernel, 
 * bspline_cuda_score_e_mse_kernel1b, because they must be performed on
 * "set by set" basis.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel1a (
	float  *dc_dv,
	float  *score,
	float3 rdims,			// Number of tiles/regions in x, y, and z
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing)		// Dimensions of a single voxel (in mm)
{
	int3   vox_coordinate;	// X, Y, Z coordinates for this voxel	
	int3   p;				// Offset of the tile in the volume (x, y and z)
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Get the X, Y, Z position of the voxel.
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);
	
		// Get the tile location of the voxel.
		p.x = vox_coordinate.x / vox_per_rgn.x;
		p.y = vox_coordinate.y / vox_per_rgn.y;
		p.z = vox_coordinate.z / vox_per_rgn.z;
	
		// Get the offset of the voxel within the tile.
		q.x = vox_coordinate.x - p.x * vox_per_rgn.x;
		q.y = vox_coordinate.y - p.y * vox_per_rgn.y;
		q.z = vox_coordinate.z - p.z * vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = qidx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}

			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
					// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);
			}	
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel1b
 *
 * This kernel calculates the dc_dv values for a given "set."  Since
 * there are a total of 64 sets, this kernel must be executed 64 times 
 * using different sidx values to completely fill the dc_dv stream.  To
 * improve performance, the score calculations are contained in a
 * separate kernel, bspline_cuda_score_e_mse_kernel1a, and calculated 
 * for the entire volume at one time.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel1b (
	float  *dc_dv,
	float  *score,
	int3   sidx,			// Current "set index" given in x, y and z
	float3 rdims,			// Number of tiles/regions in x, y, and z
	int3   sdims,           // Dimensions of the set in tiles (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	int    total_vox_per_rgn,
	float3 pix_spacing)		// Dimensions of a single voxel (in mm)
{
	int3   s;				// Offset of the tile in the set (x, y and z)
	int3   p;				// Offset of the tile in the volume (x, y and z)
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Calculate the linear "set index," which is the index of the tile in the set that contains the 
	// voxel corresponding to this thread.
	int tileIdxInSet = threadIdxInGrid / total_vox_per_rgn;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the offset of the tile within the set in the x, y, and z directions.
		s.x = tileIdxInSet % sdims.x;
		s.y = ((tileIdxInSet - s.x) / sdims.x) % sdims.y;
		s.z = ((((tileIdxInSet - s.x) / sdims.x) - s.y) / sdims.y) % sdims.z;

		// Calculate the offset of the tile in the volume, based on the set offset.
		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = qidx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
					// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel1
 *
 * As an alternative to using bspline_cuda_score_e_mse_kernel1a and
 * bspline_cuda_score_e_mse_kernel1b separately, this kernel computes
 * both the score and dc_dv stream values on a set by set basis.  Since
 * there are a total of 64 sets, this kernel must be executed 64 times 
 * using different sidx values to completely fill the score and dc_dv 
 * streams.  The performance is worse than using kernel1a and kernel1b.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel1 (
	float  *dc_dv,
	float  *score,
	int3   sidx,			// Current "set index" given in x, y and z
	float3 rdims,			// Number of tiles/regions in x, y, and z
	int3   sdims,           // Dimensions of the set in tiles (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	int    total_vox_per_rgn,
	float3 pix_spacing)		// Dimensions of a single voxel (in mm)
{
	int3   s;				// Offset of the tile in the set (x, y and z)
	int3   p;				// Offset of the tile in the volume (x, y and z)
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Calculate the linear "set index," which is the index of the tile in the set that contains the 
	// voxel corresponding to this thread.
	int tileIdxInSet = threadIdxInGrid / total_vox_per_rgn;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the offset of the tile within the set in the x, y, and z directions.
		s.x = tileIdxInSet % sdims.x;
		s.y = ((tileIdxInSet - s.x) / sdims.x) % sdims.y;
		s.z = ((((tileIdxInSet - s.x) / sdims.x) - s.y) / sdims.y) % sdims.z;

		// Calculate the offset of the tile in the volume, based on the set offset.
		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = qidx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
					// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				// The score calculation has been moved to bspline_cuda_score_e_kernel1a.
				score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel2_by_sets
 *
 * This version of kernel2 updates the gradient stream for a given set
 * of tiles.  It performs the calculation for the entire set at once,
 * which improves parallelism and therefore improves performance as
 * compared to the tile by tile implementation, which is found below.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel2_by_sets(
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int3   sidx,
	int3   sdims,
	float3 rdims,
	int3   vox_per_rgn,
	int    threads_per_tile,
	int    num_threads)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Calculate the linear "set index," which is the index of the tile in the set that contains the 
	// voxel corresponding to this thread.
	int tileIdxInSet = threadIdxInGrid / threads_per_tile;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int3 s; // Offset of the tile in the set (x, y, and z)
		int3 p; // Offset of the tile in the volume (x, y, and z)
		int m;
		int num_vox;
		int xyzOffset;
		int tileOffset;
		int pidx;
		int cidx;
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Calculate the offset of the tile within the set in the x, y, and z directions.
		s.x = tileIdxInSet % sdims.x;
		s.y = ((tileIdxInSet - s.x) / sdims.x) % sdims.y;
		s.z = ((((tileIdxInSet - s.x) / sdims.x) - s.y) / sdims.y) % sdims.z;

		// Calculate the offset of the tile in the volume, based on the set offset.
		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		// Use the offset of the tile in the volume to compute the index into the c_lut.
		pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;

		// Calculate the linear index of the control point in the range [0, 63].
		m = (threadIdxInGrid % threads_per_tile) / 3;

		// Determine if this thread corresponds to the x, y, or z coordinate,
		// where x = 0, y = 1, and z = 2.
		xyzOffset = (threadIdxInGrid % threads_per_tile) - (m * 3);

		// Calculate the index into the coefficient lookup table.
		cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels per tile.
		num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		// Calculate the offset of this tile in the dc_dv array.
		tileOffset = 3 * num_vox * tileIdxInSet;

		for(qidx = 0; qidx < num_vox - 8; qidx = qidx + 8) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+7 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+6 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+5 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+4 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+3 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+2 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+1 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel2_by_tiles
 * This version of kernel2 updates the gradient stream for a given tile.
 * Since it operates on only one tile in a set at a given time, the
 * performance is worse than bspline_cuda_score_e_mse_kernel2_by_sets.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel2_by_tiles (
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int    num_threads,
	int3   p,
	float3 rdims,
	int    offset,
	int3   vox_per_rgn,
	int    total_vox_per_rgn) // Volume of a tile in voxels)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int m;
		int num_vox;
		int xyzOffset;
		int tileOffset;
		int cidx;
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		
		// Calculate the linear index of the control point.
		m = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		xyzOffset = threadIdxInGrid - (m * 3);

		// Calculate index into coefficient texture.
		cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels in the region.
		num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		// Calculate the offset of this tile in the dc_dv array.
		tileOffset = 3 * num_vox * offset;

		/* ORIGINAL CODE: Looked at each offset serially.
		// Serial across offsets.
		for(int qidx = 0; qidx < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z); qidx++) {
			result += tex1Dfetch(tex_dc_dv, 3*qidx + offset) * tex1Dfetch(tex_q_lut, 64*qidx + m);
		}
		*/

		// NAGA: Unrolling the loop 8 times; 4 seems to work as well as 8
		// FOR_CHRIS: FIX to make sure the unrolling works with an arbitrary loop index
		for(qidx = 0; qidx < num_vox - 8; qidx = qidx + 8) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+7 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+6 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+5 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+4 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+3 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+2 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+1 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel2_by_tiles_v2
 *
 * In comparison to bspline_cuda_score_e_mse_kernel2_by_tiles_v2, this
 * kernel uses multiple threads to accumulate the influence from a tile.
 * The threads are synchronized at the end so that the partial sums can
 * be exchanged using shared memory, summed together, and saved to the
 * gradient stream.  The number of threads being used for each control
 * point must be given as an argument.  The performance is better than
 * bspline_cuda_score_e_mse_kernel2_by_tiles_v2, but the implementation
 * is still buggy.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel2_by_tiles_v2 (
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int    num_threads,
	int3   p,
	float3 rdims,
	int    offset,
	int3   vox_per_rgn,
	int    threadsPerControlPoint)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Set the number of threads being used to work on each control point.
		int tpcp = threadsPerControlPoint;

		// Calculate the linear index of the control point.
		int m = threadIdxInGrid / (threadsPerControlPoint * 3);

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		int xyzOffset = (threadIdxInGrid / threadsPerControlPoint) - (m * 3);

		// Determine the thread offset for this control point, in the range [0, threadsPerControlPoint).
		int cpThreadOffset = threadIdxInGrid % threadsPerControlPoint;

		// Calculate index into coefficient texture.
		int cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels in the region.
		int num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		// Calculate the offset of this tile in the dc_dv array.
		int tileOffset = 3 * num_vox * offset;

		for(qidx = cpThreadOffset; qidx < num_vox - (8*tpcp); qidx = qidx + (8*tpcp)) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+(7*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+(6*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+(5*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+(4*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+(3*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+(2*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+(1*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		sdata[(tpcp * threadIdxInBlock) + cpThreadOffset] = result;
		
		// Wait for the other threads in the thread block to reach this point.
		__syncthreads();

		if(cpThreadOffset == 0) {
			result = sdata[(tpcp * threadIdxInBlock) + 0] + sdata[(tpcp * threadIdxInBlock) + 1];
				
			/*
			result = 0.0;

			// Accumulate all the partial results for this control point.
			for(int i = 0; i < tpcp; i++) {
				result += sdata[(tpcp * threadIdxInBlock) + i];
			}
			*/

			// Update the gradient stream.
			grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
		}			
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel1
 *
 * This kernel is one of two used in the CUDA implementation of 
 * score_d_mse, which is intended to have reduced memory requirements.  
 * It calculuates the score and dc_dv values on a region by region basis 
 * rather than for the entire volume at once.  As a result, the score 
 * stream need only to have as many elements as there are voxels in a 
 * region.  When executing this kernel, the number of threads should be 
 * close to (but greater than) the number of voxels in a region.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel1 (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInGrid < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			// qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = threadIdxInGrid * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// diff = 0.0;
				// valid = 0;
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel1_v2
 *
 * This kernel is one of two used in the CUDA implementation of 
 * score_d_mse, which is intended to have reduced memory requirements.  
 * It calculuates the score and dc_dv values on a region by region basis 
 * rather than for the entire volume at once.  As a result, the score 
 * stream need only to have as many elements as there are voxels in a 
 * region.  When executing this kernel, the number of threads should be 
 * close to (but greater than) the number of voxels in a region.
 *
 * In comparison to bspline_cuda_score_d_mse_kernel1, this kernel 
 * computes the x, y, and z portions of each value in separate threads 
 * for increased parallelism.  The performance is worse than 
 * bspline_cuda_score_d_mse_kernel1, so this version should not be used.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel1_v2 (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	int lridx = 0;  // Linear index within the region
	int offset = 0; // x = 0, y = 1, z = 2

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInGrid < (3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the linear index of the voxel in the region. Will be in the range
		// (0, vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z - 1).
		lridx = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (lridx * 3);		

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = lridx % vox_per_rgn.x;
		q.y = ((lridx - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((lridx - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel in the volume.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;
			
			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			// qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = lridx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				// diff[threadIdxInGrid] = fixed_image[threadIdxInGrid] - m_val;
				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				if(offset == 0)
					score[lridx] = tex1Dfetch(tex_score, lridx) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[threadIdxInGrid] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + offset);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel1_v3
 *
 * This kernel is one of two used in the CUDA implementation of 
 * score_d_mse, which is intended to have reduced memory requirements.  
 * It calculuates the score and dc_dv values on a region by region basis 
 * rather than for the entire volume at once.  As a result, the score 
 * stream need only to have as many elements as there are voxels in a 
 * region.  When executing this kernel, the number of threads should be 
 * close to (but greater than) the number of voxels in a region.
 *
 * In comparison to bspline_cuda_score_d_mse_kernel2, this kernel uses 
 * shared memory to exchange data between threads to reduce the number
 * of memory accesses.  The performance is worse than 
 * bspline_cuda_score_d_mse_kernel1, so this version should not be used.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel1_v3 (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	int lridx = 0;  // Linear index within the region
	int offset = 0; // x = 0, y = 1, z = 2

	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.
	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;
	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Calculate the number of unusable threads in each block.
	int threadsLostPerBlock = threadsPerBlock - (threadsPerBlock / 3) * 3;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * (threadsPerBlock - threadsLostPerBlock)) + threadIdxInBlock;

	// Set the "write flag" to 0.
	sdata[2*(threadIdxInBlock/3)+2] = 0.0;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInBlock < (threadsPerBlock - threadsLostPerBlock) &&
		threadIdxInGrid < (3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the linear index of the voxel in the region. Will be in the range
		// (0, vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z - 1).
		lridx = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (lridx * 3);		

		// Only one out of every three threads needs to calculate the following information.
		// All other threads get the data from shared memory.
		if(offset ==  0) {

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = lridx % vox_per_rgn.x;
		q.y = ((lridx - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((lridx - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel in the volume.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;
			
			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			// qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = lridx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				
				if(offset == 0) {
					sdata[2*(threadIdxInBlock/3)] = 0.0;
					sdata[2*(threadIdxInBlock/3)+1] = 0.0;
				}
			}
			else {
					
					//-----------------------------------------------------------------
					// Compute interpolation fractions.
					//-----------------------------------------------------------------

					// Clamp and interpolate along the X axis.
					displacement_in_vox_floor.x = floor(displacement_in_vox.x);
					displacement_in_vox_round.x = round(displacement_in_vox.x);
					fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
					if(displacement_in_vox_floor.x < 0){
						displacement_in_vox_floor.x = 0;
						displacement_in_vox_round.x = 0;
						fx2 = 0.0;
					}
					else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
						displacement_in_vox_floor.x = volume_dim.x - 2;
						displacement_in_vox_round.x = volume_dim.x - 1;
						fx2 = 1.0;
					}
					fx1 = 1.0 - fx2;

					// Clamp and interpolate along the Y axis.
					displacement_in_vox_floor.y = floor(displacement_in_vox.y);
					displacement_in_vox_round.y = round(displacement_in_vox.y);
					fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
					if(displacement_in_vox_floor.y < 0){
						displacement_in_vox_floor.y = 0;
						displacement_in_vox_round.y = 0;
						fy2 = 0.0;
					}
					else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
						displacement_in_vox_floor.y = volume_dim.y - 2;
						displacement_in_vox_round.y = volume_dim.y - 1;
						fy2 = 1.0;
					}
					fy1 = 1.0 - fy2;
					
					// Clamp and intepolate along the Z axis.
					displacement_in_vox_floor.z = floor(displacement_in_vox.z);
					displacement_in_vox_round.z = round(displacement_in_vox.z);
					fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
					if(displacement_in_vox_floor.z < 0){
						displacement_in_vox_floor.z = 0;
						displacement_in_vox_round.z = 0;
						fz2 = 0.0;
					}
					else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
						displacement_in_vox_floor.z = volume_dim.z - 2;
						displacement_in_vox_round.z = volume_dim.z - 1;
						fz2 = 1.0;
					}
					fz1 = 1.0 - fz2;
					
					//-----------------------------------------------------------------
					// Compute moving image intensity using linear interpolation.
					//-----------------------------------------------------------------

					mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
					m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
					m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
					m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
					m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
					m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
					m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
					m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
					m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
					m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

					//-----------------------------------------------------------------
					// Compute intensity difference.
					//-----------------------------------------------------------------

					diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
					
					//-----------------------------------------------------------------
					// Accumulate the score.
					//-----------------------------------------------------------------
				
					score[lridx] = tex1Dfetch(tex_score, lridx) + (diff * diff);

					//-----------------------------------------------------------------
					// Compute dc_dv for this offset
					//-----------------------------------------------------------------
					
					// Compute spatial gradient using nearest neighbors.
					mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;

					// Store this data in shared memory.
					sdata[2*(threadIdxInBlock/3)] = diff;
					sdata[2*(threadIdxInBlock/3)+1] = mvr;
					sdata[2*(threadIdxInBlock/3)+2] = 1.0;
				}				
			}
		}
	}

	// Wait until all the threads in this thread block reach this point.
	__syncthreads();

	// dc_dv[threadIdxInGrid] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + offset);

	if(sdata[2*(threadIdxInBlock/3)+2] == 1.0)
		dc_dv[threadIdxInGrid] = sdata[2*(threadIdxInBlock/3)] * 
			tex1Dfetch(tex_moving_grad, (3 * (int)sdata[2*(threadIdxInBlock/3)+1]) + offset);
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel2
 *
 * This kernel is the second of two used in the CUDA implementation of
 * score_d_mse.  It calculates the values for the gradient stream on 
 * a tile by tile basis.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel2 (
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int    num_threads,
	int3   p,
	float3 rdims,
	int3   vox_per_rgn)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{	
		int m;
		int offset;
		int cidx;
		int qidx;
		int num_vox;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		
		// Calculate the linear index of the control point.
		m = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (m * 3);

		// Calculate index into coefficient texture.
		cidx = tex1Dfetch(tex_c_lut, 64*pidx + m) * 3;

		// Calculate the number of voxels in the region.
		num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		/* ORIGINAL CODE: Looked at each offset serially.
		// Serial across offsets.
		for(int qidx = 0; qidx < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z); qidx++) {
			result += tex1Dfetch(tex_dc_dv, 3*qidx + offset) * tex1Dfetch(tex_q_lut, 64*qidx + m);
		}
		*/

		// NAGA: Unrolling the loop 8 times; 4 seems to work as well as 8
		// FOR_CHRIS: FIX to make sure the unrolling works with an arbitrary loop index
		for(qidx = 0; qidx < num_vox - 8; qidx = qidx + 8) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+7 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+6 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+5 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+4 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+3 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+2 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+1 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		grad[cidx + offset] = tex1Dfetch(tex_grad, cidx + offset) + result;
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel2_v2
 *
 * This kernel is the second of two used in the CUDA implementation of
 * score_d_mse.  It calculates the values for the gradient stream on 
 * a tile by tile basis.
 *
 * In comparison to bspline_cuda_score_d_mse_kernel2, this kernel uses
 * multiple threads to accumulate the influence from a tile.  The
 * threads are synchronized at the end so that the partial sums can be
 * exchanged using shared memory, totaled, and accumulated into the
 * gradient stream.  The number of threads being used for each control
 * point must be given as an argument.  The performance is better than 
 * bspline_cuda_score_d_mse_kernel2, although the implementation is
 * still buggy.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel2_v2 (
	float* grad,
	int    num_threads,
	int3   p,
	float3 rdims,
	int3   vox_per_rgn,
	int    threadsPerControlPoint)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Set the number of threads being used to work on each control point.
		int tpcp = threadsPerControlPoint;

		// Calculate the linear index of the control point.
		int m = threadIdxInGrid / (threadsPerControlPoint * 3);

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		int xyzOffset = (threadIdxInGrid / threadsPerControlPoint) - (m * 3);

		// Determine the thread offset for this control point, in the range [0, threadsPerControlPoint).
		int cpThreadOffset = threadIdxInGrid % threadsPerControlPoint;

		// Calculate index into coefficient texture.
		int cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels in the region.
		int num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		for(qidx = cpThreadOffset; qidx < num_vox - (8*tpcp); qidx = qidx + (8*tpcp)) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+(7*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+(6*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+(5*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+(4*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+(3*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+(2*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+(1*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		sdata[(tpcp * threadIdxInBlock) + cpThreadOffset] = result;
		
		// Wait for the other threads in the thread block to reach this point.
		__syncthreads();

		if(cpThreadOffset == 0) {
			result = 0.0;

			// Accumulate all the partial results for this control point.
			for(int i = 0; i < tpcp; i++) {
				result += sdata[(tpcp * threadIdxInBlock) + i];
			}
			
			// Update the gradient stream.
			grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
		}			
	}
}

/***********************************************************************
 * bspline_cuda_compute_dxyz_kernel
 *
 * This kernel computes the displacement values in the x, y, and 
 * z directions.
 ***********************************************************************/
__global__ void bspline_cuda_compute_dxyz_kernel(
	int   *c_lut,
	float *q_lut,
	float *coeff,
	int3 volume_dim,
	int3 vox_per_rgn,
	float3 rdims,
	float *dx,
	float *dy,
	float *dz
	)
{
	int3 vox_coordinate;	// X, Y, Z coordinates for this voxel	
	int3 p;				    // Tile index.
	int3 q;				    // Offset within tile.
	int pidx;				// Index into c_lut.
	int qidx;				// Index into q_lut.
	int cidx;				// Index into the coefficient table.
	int* prow;				// First element in the correct row in c_lut.
	float P;				
	float3 N;				// Multiplier values.		
	float3 output;			// Output values.

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Get the X, Y, Z position of the voxel.
		// vox_coordinate.z = floor(threadIdxInGrid / (volume_dim.x * volume_dim.y));
		// vox_coordinate.y = floor((threadIdxInGrid - vox_coordinate.z * (volume_dim.x * volume_dim.y)) / volume_dim.x);
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);
			
		// Get the tile location of the voxel.
		p.x = vox_coordinate.x / vox_per_rgn.x;
		p.y = vox_coordinate.y / vox_per_rgn.y;
		p.z = vox_coordinate.z / vox_per_rgn.z;
				
		// Get the offset of the voxel within the tile.
		q.x = vox_coordinate.x - p.x * vox_per_rgn.x;
		q.y = vox_coordinate.y - p.y * vox_per_rgn.y;
		q.z = vox_coordinate.z - p.z * vox_per_rgn.z;
				
		// Use the tile location of the voxel to compute the index into the c_lut.
		pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		prow = &c_lut[pidx*64];
		pidx = pidx * 64;

		// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
		qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
		// qrow = &q_lut[qidx*64];
		qidx = qidx * 64;

		// Initialize output values.
		output.x = 0.0;
		output.y = 0.0;
		output.z = 0.0;

		for(int k = 0; k < 64; k++)
		{
			// Calculate the index into the coefficients array.
			cidx = 3 * prow[k];
			// cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
			
			// Fetch the values for P, Ni, Nj, and Nk.
			// P = qrow[k];
			P  = tex1Dfetch(tex_q_lut, qidx + k); 
			N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
			N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
			N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

			// Update the output (v) values.
			output.x += P * N.x;
			output.y += P * N.y;
			output.z += P * N.z;
		}

		// Save the calculated values to the output streams.
		dx[threadIdxInGrid] = output.x;
		dy[threadIdxInGrid] = output.y;
		dz[threadIdxInGrid] = output.z;
	}
}

/***********************************************************************
 * bspline_cuda_compute_diff_kernel
 *
 * This kernel computes the intensity difference between the voxels
 * in the moving and fixed images.
 ***********************************************************************/
__global__ void bspline_cuda_compute_diff_kernel (
	float* fixed_image,
	float* moving_image,
	float* dx,
	float* dy,
	float* dz,
	float* diff,
	int*   valid_voxels,
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// x, y, z coordinates for the image origin
	float3 pix_spacing,		// Dimensions of a single voxel in millimeters
	float3 img_offset)		// Offset corresponding to the region of interest
{	

	int3   vox_coordinate;
	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	int3   displacement_in_vox_floor;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Ensure that the thread index corresponds to a voxel in the volume before continuing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{ 
		// Get the x, y, z position of the voxel.
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);

		// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
		distance_from_image_origin.x = img_origin.x + (pix_spacing.x * vox_coordinate.x);
		distance_from_image_origin.y = img_origin.y + (pix_spacing.y * vox_coordinate.y);
		distance_from_image_origin.z = img_origin.z + (pix_spacing.z * vox_coordinate.z);
		
		// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
		displacement_in_mm.x = distance_from_image_origin.x + tex1Dfetch(tex_dx, threadIdxInGrid); //dx[threadIdxInGrid];
		displacement_in_mm.y = distance_from_image_origin.y + tex1Dfetch(tex_dy, threadIdxInGrid); //dy[threadIdxInGrid];
		displacement_in_mm.z = distance_from_image_origin.z + tex1Dfetch(tex_dz, threadIdxInGrid); //dz[threadIdxInGrid];

		// Calculate the displacement value in terms of voxels.
		displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
		displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
		displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

		// Check if the displaced voxel lies outside the region of interest.
		if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
			(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
			(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
			diff[threadIdxInGrid] = 0.0;
			valid_voxels[threadIdxInGrid] = 0;
		}
		else {

			// Clamp and interpolate along the X axis.
			displacement_in_vox_floor.x = (int)floor(displacement_in_vox.x);
			fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
			if(displacement_in_vox_floor.x < 0){
				displacement_in_vox_floor.x = 0;
				fx2 = 0.0;
			}
			else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
				displacement_in_vox_floor.x = volume_dim.x - 2;
				fx2 = 1.0;
			}
			fx1 = 1.0 - fx2;
			
			// Clamp and interpolate along the Y axis.
			displacement_in_vox_floor.y = (int)floor(displacement_in_vox.y);
			fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
			if(displacement_in_vox_floor.y < 0){
				displacement_in_vox_floor.y = 0;
				fy2 = 0.0;
			}
			else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
				displacement_in_vox_floor.y = volume_dim.y - 2;
				fy2 = 1.0;
			}
			fy1 = 1.0 - fy2;
			
			// Clamp and intepolate along the Z axis.
			displacement_in_vox_floor.z = (int)floor(displacement_in_vox.z);
			fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
			if(displacement_in_vox_floor.z < 0){
				displacement_in_vox_floor.z = 0;
				fz2 = 0.0;
			}
			else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
				displacement_in_vox_floor.z = volume_dim.z - 2;
				fz2 = 1.0;
			}
			fz1 = 1.0 - fz2;
			
			// Compute moving image intensity using linear interpolation.
			mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
			/*
			m_x1y1z1 = fx1 * fy1 * fz1 * moving_image[mvf];
			m_x2y1z1 = fx2 * fy1 * fz1 * moving_image[mvf + 1];
			m_x1y2z1 = fx1 * fy2 * fz1 * moving_image[mvf + volume_dim.x];
			m_x2y2z1 = fx2 * fy2 * fz1 * moving_image[mvf + volume_dim.x + 1];
			m_x1y1z2 = fx1 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x];
			m_x2y1z2 = fx2 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + 1];
			m_x1y2z2 = fx1 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x];
			m_x2y2z2 = fx2 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1];
			*/
			m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
			m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
			m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
			m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
			m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
			m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
			m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
			m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
			m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

			// Compute intensity difference.
			// diff[threadIdxInGrid] = fixed_image[threadIdxInGrid] - m_val;
			diff[threadIdxInGrid] = tex1Dfetch(tex_fixed_image, threadIdxInGrid) - m_val;
			valid_voxels[threadIdxInGrid] = 1;
		}
	}
}

/***********************************************************************
 * bspline_cuda_compute_dc_dv_kernel
 *
 * This kernel computes the dc_dv values used to update the control knot
 * coefficients.
 ***********************************************************************/
__global__ void bspline_cuda_compute_dc_dv_kernel (
	float  *fixed_image,
	float  *moving_image,
	float  *moving_grad,
	int    *c_lut,
	float  *q_lut,
	float  *dx,
	float  *dy,
	float  *dz,
	float  *diff,
	float  *dc_dv_x,
	float  *dc_dv_y,
	float  *dc_dv_z,
	// float  *grad,
	int    *valid_voxels,
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	int3   vox_per_rgn,
	float3 rdims,
	float3 img_origin,		// x, y, z coordinates for the image origin
	float3 pix_spacing,		// Dimensions of a single voxel in millimeters
	float3 img_offset)		// Offset corresponding to the region of interest
{	
	int3   vox_coordinate;
	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  mvr;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Ensure that the thread index corresponds to a voxel in the volume before continuing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{ 
		// Get the x, y, z position of the voxel.
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);

		// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
		distance_from_image_origin.x = img_origin.x + (pix_spacing.x * vox_coordinate.x);
		distance_from_image_origin.y = img_origin.y + (pix_spacing.y * vox_coordinate.y);
		distance_from_image_origin.z = img_origin.z + (pix_spacing.z * vox_coordinate.z);
		
		// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
		displacement_in_mm.x = distance_from_image_origin.x + tex1Dfetch(tex_dx, threadIdxInGrid); //dx[threadIdxInGrid];
		displacement_in_mm.y = distance_from_image_origin.y + tex1Dfetch(tex_dy, threadIdxInGrid); //dy[threadIdxInGrid];
		displacement_in_mm.z = distance_from_image_origin.z + tex1Dfetch(tex_dz, threadIdxInGrid); //dz[threadIdxInGrid];

		// Calculate the displacement value in terms of voxels.
		displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
		displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
		displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

		/*
		// Get the tile location of the voxel.
		p.x = vox_coordinate.x / vox_per_rgn.x;
		p.y = vox_coordinate.y / vox_per_rgn.y;
		p.z = vox_coordinate.z / vox_per_rgn.z;
				
		// Get the offset of the voxel within the tile.
		q.x = vox_coordinate.x - p.x * vox_per_rgn.x;
		q.y = vox_coordinate.y - p.y * vox_per_rgn.y;
		q.z = vox_coordinate.z - p.z * vox_per_rgn.z;
				
		// Use the tile location of the voxel to compute the index into the c_lut.
		pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		prow = &c_lut[pidx*64];

		// Use the offset if the voxel to compute the index into the multiplier LUT or q_lut.
		qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
		qrow = &q_lut[qidx*64];
		*/

		// Check if the displaced voxel lies outside the region of interest.
		if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
			(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
			(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
			dc_dv_x[threadIdxInGrid] = 0.0;
			dc_dv_y[threadIdxInGrid] = 0.0;
			dc_dv_z[threadIdxInGrid] = 0.0;
		}
		else {

			// Clamp and interpolate along the X axis.
			displacement_in_vox_floor.x = floor(displacement_in_vox.x);
			displacement_in_vox_round.x = round(displacement_in_vox.x);
			if(displacement_in_vox_floor.x < 0){
				displacement_in_vox_floor.x = 0;
				displacement_in_vox_round.x = 0;
			}
			else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
				displacement_in_vox_floor.x = volume_dim.x - 2;
				displacement_in_vox_round.x = volume_dim.x - 1;
			}
			
			// Clamp and interpolate along the Y axis.
			displacement_in_vox_floor.y = floor(displacement_in_vox.y);
			displacement_in_vox_round.y = round(displacement_in_vox.y);
			if(displacement_in_vox_floor.y < 0){
				displacement_in_vox_floor.y = 0;
				displacement_in_vox_round.y = 0;
			}
			else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
				displacement_in_vox_floor.y = volume_dim.y - 2;
				displacement_in_vox_round.y = volume_dim.y - 1;
			}
			
			// Clamp and intepolate along the Z axis.
			displacement_in_vox_floor.z = floor(displacement_in_vox.z);
			displacement_in_vox_round.z = round(displacement_in_vox.z);
			if(displacement_in_vox_floor.z < 0){
				displacement_in_vox_floor.z = 0;
				displacement_in_vox_round.z = 0;
			}
			else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
				displacement_in_vox_floor.z = volume_dim.z - 2;
				displacement_in_vox_round.z = volume_dim.z - 1;
			}

			// Compute spatial gradient using nearest neighbors.
			mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
			dc_dv_x[threadIdxInGrid] = diff[threadIdxInGrid] * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0); //moving_grad[(3 * (int)mvr) + 0];
			dc_dv_y[threadIdxInGrid] = diff[threadIdxInGrid] * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1); //moving_grad[(3 * (int)mvr) + 1];
			dc_dv_z[threadIdxInGrid] = diff[threadIdxInGrid] * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2); //moving_grad[(3 * (int)mvr) + 2];
			
			/*
		    for (int i = 0; i < 64; i++) {
				cidx = 3 * prow[i];
				grad[cidx+0] += dc_dv.x * qrow[i];
				grad[cidx+1] += dc_dv.y * qrow[i];
				grad[cidx+2] += dc_dv.z * qrow[i];
			}
			*/
		}
	}
}

/***********************************************************************
 * sum_reduction_kernel
 *
 * This kernel will reduce a stream to a single value.  It will work for
 * a stream with an arbitrary number of elements.  It is the same as 
 * bspline_cuda_compute_score_kernel, with the exception that it assumes
 * all values in the stream are valid and should be included in the final
 * reduced value.
 ***********************************************************************/
__global__ void sum_reduction_kernel(
  float *idata, 
  float *odata, 
  int   num_elems)
{
  // Shared memory is allocated on a per block basis.  Therefore, only allocate 
  // (sizeof(data) * blocksize) memory when calling the kernel.
  extern __shared__ float sdata[];
  
  // Calculate the index of the thread block in the grid.
  int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
  
  // Calculate the total number of threads in each thread block.
  int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);
  
  // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
  int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
  
  // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
  int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

  // Load data into shared memory.
  if(threadIdxInGrid >= num_elems)
    sdata[threadIdxInBlock] = 0.0;
  else 
    sdata[threadIdxInBlock] = idata[threadIdxInGrid];

  // Wait for all threads in the block to reach this point.
  __syncthreads();
  
  // Perform the reduction in shared memory.  Stride over the block and reduce
  // parts until it is down to a single value (stored in sdata[0]).
  for(unsigned int s = threadsPerBlock / 2; s > 0; s >>= 1) {
    if (threadIdxInBlock < s) {
      sdata[threadIdxInBlock] += sdata[threadIdxInBlock + s];
    }

    // Wait for all threads to complete this stride.
    __syncthreads();
  }
  
  // Write the result for this block back to global memory.
  if(threadIdxInBlock == 0) {
	  odata[threadIdxInGrid] = sdata[0];
  }
}

/***********************************************************************
 * bspline_cuda_compute_score_kernel
 *
 * This kernel reduces the score stream to a single value.  It will work
 * for an aribtrary stream size, and also checks a flag for each element
 * to determine whether or not it is "valid" before adding it to the
 * final sum.
 ***********************************************************************/
__global__ void bspline_cuda_compute_score_kernel(
  float *idata, 
  float *odata, 
  int   *valid_voxels, 
  int   num_elems)
{
  // Shared memory is allocated on a per block basis.  Therefore, only allocate 
  // (sizeof(data) * blocksize) memory when calling the kernel.
  extern __shared__ float sdata[]; 
  
  // Calculate the index of the thread block in the grid.
  int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
  
  // Calculate the total number of threads in each thread block.
  int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);
  
  // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
  int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
  
  // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
  int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

  // Load data into shared memory.
  if(threadIdxInGrid >= num_elems || valid_voxels[threadIdxInGrid] == 0)
    sdata[threadIdxInBlock] = 0.0;
  else 
    sdata[threadIdxInBlock] = idata[threadIdxInGrid] * idata[threadIdxInGrid];

  // Wait for all threads in the block to reach this point.
  __syncthreads();
  
  // Perform the reduction in shared memory.  Stride over the block and reduce
  // parts until it is down to a single value (stored in sdata[0]).
  for(unsigned int s = threadsPerBlock / 2; s > 0; s >>= 1) {
    if (threadIdxInBlock < s) {
      sdata[threadIdxInBlock] += sdata[threadIdxInBlock + s];
    }

    // Wait for all threads to complete this stride.
    __syncthreads();
  }
  
  // Write the result for this block back to global memory.
  if(threadIdxInBlock == 0) {
	  odata[threadIdxInGrid] = sdata[0];
  }
}

/***********************************************************************
 * sum_reduction_last_step_kernel
 *
 * This kernel sums together the remaining partial sums that are created
 * by the other sum reduction kernels.
 ***********************************************************************/
__global__ void sum_reduction_last_step_kernel(
	float *idata,
	float *odata,
	int   num_elems)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	if(threadIdxInGrid == 0) {
	
		float sum = 0.0;
		
		for(int i = 0; i < num_elems; i += threadsPerBlock) {
			sum += idata[i];
		}

		odata[0] = sum;
	}
}

/***********************************************************************
 * bspline_cuda_update_grad_kernel
 *
 * This kernel updates each of the gradient values before the final
 * sum reduction of the gradient stream.
 ***********************************************************************/
__global__ void bspline_cuda_update_grad_kernel(
	float *grad,
	int num_vox,
	int num_elems)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	if(threadIdxInGrid < num_elems) {
		grad[threadIdxInGrid] = 2 * tex1Dfetch(tex_grad, threadIdxInGrid) / num_vox;
	}
}

/***********************************************************************
 * bspline_cuda_compute_grad_mean_kernel
 *
 * This kernel computes the value of grad_mean from the gradient stream.
 ***********************************************************************/
__global__ void bspline_cuda_compute_grad_mean_kernel(
	float *idata,
	float *odata,
	int num_elems)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Load data into shared memory.
	if(threadIdxInGrid >= num_elems)
		sdata[threadIdxInBlock] = 0.0;
	else 
		sdata[threadIdxInBlock] = idata[threadIdxInGrid];

	// Wait for all threads in the block to reach this point.
	__syncthreads();

	// Perform the reduction in shared memory.  Stride over the block and reduce
	// parts until it is down to a single value (stored in sdata[0]).
	for(unsigned int s = threadsPerBlock / 2; s > 0; s >>= 1) {
		if (threadIdxInBlock < s) {
			sdata[threadIdxInBlock] += sdata[threadIdxInBlock + s];
		}

		// Wait for all threads to complete this stride.
		__syncthreads();
	}

	// Write the result for this block back to global memory.
	if(threadIdxInBlock == 0) {
		odata[threadIdxInGrid] = sdata[0];
	}
}


/***********************************************************************
 * bspline_cuda_compute_grad_norm_kernel
 *
 * This kernel computes the value of grad_norm from the gradient stream.
 ***********************************************************************/
__global__ void bspline_cuda_compute_grad_norm_kernel(
	float *idata,
	float *odata,
	int num_elems)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Load data into shared memory.
	if(threadIdxInGrid >= num_elems)
		sdata[threadIdxInBlock] = 0.0;
	else 
		sdata[threadIdxInBlock] = fabs(idata[threadIdxInGrid]);

	// Wait for all threads in the block to reach this point.
	__syncthreads();

	// Perform the reduction in shared memory.  Stride over the block and reduce
	// parts until it is down to a single value (stored in sdata[0]).
	for(unsigned int s = threadsPerBlock / 2; s > 0; s >>= 1) {
		if (threadIdxInBlock < s) {
			sdata[threadIdxInBlock] += sdata[threadIdxInBlock + s];
		}

		// Wait for all threads to complete this stride.
		__syncthreads();
	}

	// Write the result for this block back to global memory.
	if(threadIdxInBlock == 0) {
		odata[threadIdxInGrid] = sdata[0];
	}
}

/***********************************************************************
 * bspline_cuda_initialize_g
 *
 * Initialize the GPU to execute bspline_cuda_score_g_mse().
 ***********************************************************************/
void bspline_cuda_initialize_g(
			       Volume *fixed,
			       Volume *moving,
			       Volume *moving_grad,
			       BSPLINE_Xform *bxf,
			       BSPLINE_Parms *parms)
{
#if defined (commentout)
    printf("Initializing CUDA (g) ... ");
#endif

    unsigned int total_bytes = 0;

    // Copy the fixed image to the GPU.
    if(cudaMalloc((void**)&gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for fixed image");
    if(cudaMemcpy(gpu_fixed_image, fixed->img, fixed->npix * fixed->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
	checkCUDAError("Failed to copy fixed image to GPU");
    if(cudaBindTexture(0, tex_fixed_image, gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_fixed_image to linear memory");
    total_bytes += fixed->npix * fixed->pix_size;

    // Copy the moving image to the GPU.
    if(cudaMalloc((void**)&gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for moving image");
    if(cudaMemcpy(gpu_moving_image, moving->img, moving->npix * moving->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
	checkCUDAError("Failed to copy moving image to GPU");
    if(cudaBindTexture(0, tex_moving_image, gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_moving_image to linear memory");
    total_bytes += moving->npix * moving->pix_size;

    // Copy the moving gradient to the GPU.
    if(cudaMalloc((void**)&gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for moving gradient");
    if(cudaMemcpy(gpu_moving_grad, moving_grad->img, moving_grad->npix * moving_grad->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
	checkCUDAError("Failed to copy moving gradient to GPU");
    if(cudaBindTexture(0, tex_moving_grad, gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_moving_grad to linear memory");
    total_bytes += moving_grad->npix * moving_grad->pix_size;

    // Allocate memory for the coefficient LUT on the GPU.  The LUT will be copied to the
    // GPU each time bspline_cuda_score_d_mse is called.
    coeff_mem_size = sizeof(float) * bxf->num_coeff;
    if(cudaMalloc((void**)&gpu_coeff, coeff_mem_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for coefficient LUT");
    if(cudaBindTexture(0, tex_coeff, gpu_coeff, coeff_mem_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_coeff to linear memory");
    total_bytes += coeff_mem_size;

    // Allocate memory to hold the calculated dc_dv values.
    dc_dv_mem_size = 3 
	    * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2]
	    * bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float);
#if defined (commentout)
    printf ("vox_per_rgn (%d,%d,%d), rdim (%d,%d,%d), bytes %d\n", 
	    bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2],
	    bxf->rdims[0], bxf->rdims[1], bxf->rdims[2],
	    dc_dv_mem_size);
#endif
    if(cudaMalloc((void**)&gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for the dc_dv stream on GPU");
    if(cudaBindTexture(0, tex_dc_dv, gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_dc_dv to linear memory");
    bspline_cuda_clear_dc_dv();
    total_bytes += dc_dv_mem_size;

    // Allocate memory to hold the calculated score values.
    score_mem_size = fixed->npix * sizeof(float);
    if(cudaMalloc((void**)&gpu_score, score_mem_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for the score stream on GPU");
    if(cudaBindTexture(0, tex_score, gpu_score, score_mem_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_score to linear memory");
    total_bytes += score_mem_size;

    // Allocate memory to hold the gradient values.
    if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for the grad stream on GPU");
    if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
	checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
    if(cudaBindTexture(0, tex_grad, gpu_grad, coeff_mem_size) != cudaSuccess)
	checkCUDAError("Failed to bind tex_grad to linear memory");
    total_bytes += 2 * coeff_mem_size;

#if defined (commentout)
    printf("DONE!\n");
    printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
#endif
}

/***********************************************************************
 * bspline_cuda_initialize_f
 *
 * Initialize the GPU to execute bspline_cuda_score_f_mse().
 ***********************************************************************/
void bspline_cuda_initialize_f(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
#if defined (commentout)
	printf("Initializing CUDA... ");
#endif

	unsigned int total_bytes = 0;

	// Copy the fixed image to the GPU.
	if(cudaMalloc((void**)&gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for fixed image");
	if(cudaMemcpy(gpu_fixed_image, fixed->img, fixed->npix * fixed->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy fixed image to GPU");
	if(cudaBindTexture(0, tex_fixed_image, gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_fixed_image to linear memory");
	total_bytes += fixed->npix * fixed->pix_size;

	// Copy the moving image to the GPU.
	if(cudaMalloc((void**)&gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving image");
	if(cudaMemcpy(gpu_moving_image, moving->img, moving->npix * moving->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving image to GPU");
	if(cudaBindTexture(0, tex_moving_image, gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_image to linear memory");
	total_bytes += moving->npix * moving->pix_size;

	// Copy the moving gradient to the GPU.
	if(cudaMalloc((void**)&gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving gradient");
	if(cudaMemcpy(gpu_moving_grad, moving_grad->img, moving_grad->npix * moving_grad->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving gradient to GPU");
	if(cudaBindTexture(0, tex_moving_grad, gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_grad to linear memory");
	total_bytes += moving_grad->npix * moving_grad->pix_size;

	// Allocate memory for the coefficient LUT on the GPU.  The LUT will be copied to the
	// GPU each time bspline_cuda_score_f_mse is called.
	coeff_mem_size = sizeof(float) * bxf->num_coeff;
	if(cudaMalloc((void**)&gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for coefficient LUT");
	if(cudaBindTexture(0, tex_coeff, gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_coeff to linear memory");
	total_bytes += coeff_mem_size;

	// Copy the multiplier LUT to the GPU.
	size_t q_lut_mem_size = sizeof(float)
		* bxf->vox_per_rgn[0]
		* bxf->vox_per_rgn[1]
		* bxf->vox_per_rgn[2]
		* 64;
	if(cudaMalloc((void**)&gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for multiplier LUT");
	if(cudaMemcpy(gpu_q_lut, bxf->q_lut, q_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy multiplier LUT to GPU");
	if(cudaBindTexture(0, tex_q_lut, gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_q_lut to linear memory");
	total_bytes += q_lut_mem_size;

	// Copy the index LUT to the GPU.
	size_t c_lut_mem_size = sizeof(int) 
		* bxf->rdims[0] 
		* bxf->rdims[1] 
		* bxf->rdims[2] 
		* 64;
	if(cudaMalloc((void**)&gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for index LUT");
	if(cudaMemcpy(gpu_c_lut, bxf->c_lut, c_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy index LUT to GPU");
	if(cudaBindTexture(0, tex_c_lut, gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_c_lut to linear memory");
	total_bytes += c_lut_mem_size;

	// Allocate memory to hold the calculated dc_dv values.
	dc_dv_mem_size = 3 * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2]
		* bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv stream on GPU");
	if(cudaBindTexture(0, tex_dc_dv, gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv to linear memory");
	bspline_cuda_clear_dc_dv();
	total_bytes += dc_dv_mem_size;

	/*
	dc_dv_mem_size = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2]
		* bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_dc_dv_x, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv_x stream on GPU");
	if(cudaMalloc((void**)&gpu_dc_dv_y, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv_y stream on GPU");
	if(cudaMalloc((void**)&gpu_dc_dv_z, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv_z stream on GPU");
	if(cudaBindTexture(0, tex_dc_dv_x, gpu_dc_dv_x, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv_x to linear memory");
	if(cudaBindTexture(0, tex_dc_dv_y, gpu_dc_dv_y, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv_y to linear memory");
	if(cudaBindTexture(0, tex_dc_dv_z, gpu_dc_dv_z, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv_z to linear memory");
	if(cudaMemset(gpu_dc_dv_x, 0, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to clear the dc_dv_x stream on GPU\n");
	if(cudaMemset(gpu_dc_dv_y, 0, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to clear the dc_dv_y stream on GPU\n");
	if(cudaMemset(gpu_dc_dv_z, 0, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to clear the dc_dv_z stream on GPU\n");
	total_bytes += 3 * dc_dv_mem_size;
	*/

	// Allocate memory to hold the calculated score values.
	score_mem_size = fixed->npix * sizeof(float);
	if(cudaMalloc((void**)&gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the score stream on GPU");
	if(cudaBindTexture(0, tex_score, gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_score to linear memory");
	total_bytes += score_mem_size;

	
	// Allocate memory to hold the diff values.
	if(cudaMalloc((void**)&gpu_diff, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the diff stream on GPU");
	if(cudaBindTexture(0, tex_diff, gpu_diff, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_diff to linear memory");
	total_bytes += score_mem_size;

	// Allocate memory to hold the mvr values.
	if(cudaMalloc((void**)&gpu_mvr, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the mvr stream on GPU");
	if(cudaBindTexture(0, tex_mvr, gpu_mvr, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_mvr to linear memory");
	total_bytes += score_mem_size;

	// Allocate memory to hold the gradient values.
	if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad stream on GPU");
	if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
	if(cudaBindTexture(0, tex_grad, gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_grad to linear memory");
	total_bytes += 2 * coeff_mem_size;

#if defined (commentout)
	printf("DONE!\n");
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
#endif
}

/***********************************************************************
 * bspline_cuda_initialize_e_v2
 *
 * Initialize the GPU to execute bspline_cuda_score_e_mse_v2().
 ***********************************************************************/
void bspline_cuda_initialize_e_v2(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	printf("Initializing CUDA... ");

	unsigned int total_bytes = 0;

	int num_tiles = (int)(ceil(bxf->rdims[0] / 4.0) * ceil(bxf->rdims[1] / 4.0) * ceil(bxf->rdims[2] / 4.0));

	// Copy the fixed image to the GPU.
	if(cudaMalloc((void**)&gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for fixed image");
	if(cudaMemcpy(gpu_fixed_image, fixed->img, fixed->npix * fixed->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy fixed image to GPU");
	if(cudaBindTexture(0, tex_fixed_image, gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_fixed_image to linear memory");
	total_bytes += fixed->npix * fixed->pix_size;

	// Copy the moving image to the GPU.
	if(cudaMalloc((void**)&gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving image");
	if(cudaMemcpy(gpu_moving_image, moving->img, moving->npix * moving->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving image to GPU");
	if(cudaBindTexture(0, tex_moving_image, gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_image to linear memory");
	total_bytes += moving->npix * moving->pix_size;

	// Copy the moving gradient to the GPU.
	if(cudaMalloc((void**)&gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving gradient");
	if(cudaMemcpy(gpu_moving_grad, moving_grad->img, moving_grad->npix * moving_grad->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving gradient to GPU");
	if(cudaBindTexture(0, tex_moving_grad, gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_grad to linear memory");
	total_bytes += moving_grad->npix * moving_grad->pix_size;

	// Allocate memory for the coefficient LUT on the GPU.  The LUT will be copied to the
	// GPU each time bspline_cuda_score_d_mse is called.
	coeff_mem_size = sizeof(float) * bxf->num_coeff;
	if(cudaMalloc((void**)&gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for coefficient LUT");
	if(cudaBindTexture(0, tex_coeff, gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_coeff to linear memory");
	total_bytes += coeff_mem_size;

	// Copy the multiplier LUT to the GPU.
	size_t q_lut_mem_size = sizeof(float)
		* bxf->vox_per_rgn[0]
		* bxf->vox_per_rgn[1]
		* bxf->vox_per_rgn[2]
		* 64;
	if(cudaMalloc((void**)&gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for multiplier LUT");
	if(cudaMemcpy(gpu_q_lut, bxf->q_lut, q_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy multiplier LUT to GPU");
	if(cudaBindTexture(0, tex_q_lut, gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_q_lut to linear memory");
	total_bytes += q_lut_mem_size;

	// Copy the index LUT to the GPU.
	size_t c_lut_mem_size = sizeof(int) 
		* bxf->rdims[0] 
		* bxf->rdims[1] 
		* bxf->rdims[2] 
		* 64;
	if(cudaMalloc((void**)&gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for index LUT");
	if(cudaMemcpy(gpu_c_lut, bxf->c_lut, c_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy index LUT to GPU");
	if(cudaBindTexture(0, tex_c_lut, gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_c_lut to linear memory");
	total_bytes += c_lut_mem_size;

	// Allocate memory to hold the calculated dc_dv values.
	dc_dv_mem_size = num_tiles * 3 * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv stream on GPU");
	if(cudaBindTexture(0, tex_dc_dv, gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv to linear memory");
	total_bytes += dc_dv_mem_size;

	// Allocate memory to hold the calculated score values.
	score_mem_size = fixed->npix * fixed->pix_size;
	if(cudaMalloc((void**)&gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the score stream on GPU");
	if(cudaBindTexture(0, tex_score, gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_score to linear memory");
	total_bytes += score_mem_size;

	// Allocate memory to hold the gradient values.
	if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad stream on GPU");
	if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
	if(cudaBindTexture(0, tex_grad, gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_grad to linear memory");
	total_bytes += 2 * coeff_mem_size;

	printf("DONE!\n");
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
}

/***********************************************************************
 * bspline_cuda_initialize_e
 *
 * Initialize the GPU to execute bspline_cuda_score_e_mse().
 ***********************************************************************/
void bspline_cuda_initialize_e(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	printf("Initializing CUDA... ");

	unsigned int total_bytes = 0;

	int num_tiles = (int)(ceil(bxf->rdims[0] / 4.0) * ceil(bxf->rdims[1] / 4.0) * ceil(bxf->rdims[2] / 4.0));

	// Copy the fixed image to the GPU.
	if(cudaMalloc((void**)&gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for fixed image");
	if(cudaMemcpy(gpu_fixed_image, fixed->img, fixed->npix * fixed->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy fixed image to GPU");
	if(cudaBindTexture(0, tex_fixed_image, gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_fixed_image to linear memory");
	total_bytes += fixed->npix * fixed->pix_size;

	// Copy the moving image to the GPU.
	if(cudaMalloc((void**)&gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving image");
	if(cudaMemcpy(gpu_moving_image, moving->img, moving->npix * moving->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving image to GPU");
	if(cudaBindTexture(0, tex_moving_image, gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_image to linear memory");
	total_bytes += moving->npix * moving->pix_size;

	// Copy the moving gradient to the GPU.
	if(cudaMalloc((void**)&gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving gradient");
	if(cudaMemcpy(gpu_moving_grad, moving_grad->img, moving_grad->npix * moving_grad->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving gradient to GPU");
	if(cudaBindTexture(0, tex_moving_grad, gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_grad to linear memory");
	total_bytes += moving_grad->npix * moving_grad->pix_size;

	// Allocate memory for the coefficient LUT on the GPU.  The LUT will be copied to the
	// GPU each time bspline_cuda_score_d_mse is called.
	coeff_mem_size = sizeof(float) * bxf->num_coeff;
	if(cudaMalloc((void**)&gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for coefficient LUT");
	if(cudaBindTexture(0, tex_coeff, gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_coeff to linear memory");
	total_bytes += coeff_mem_size;

	// Copy the multiplier LUT to the GPU.
	size_t q_lut_mem_size = sizeof(float)
		* bxf->vox_per_rgn[0]
		* bxf->vox_per_rgn[1]
		* bxf->vox_per_rgn[2]
		* 64;
	if(cudaMalloc((void**)&gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for multiplier LUT");
	if(cudaMemcpy(gpu_q_lut, bxf->q_lut, q_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy multiplier LUT to GPU");
	if(cudaBindTexture(0, tex_q_lut, gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_q_lut to linear memory");
	total_bytes += q_lut_mem_size;

	// Copy the index LUT to the GPU.
	size_t c_lut_mem_size = sizeof(int) 
		* bxf->rdims[0] 
		* bxf->rdims[1] 
		* bxf->rdims[2] 
		* 64;
	if(cudaMalloc((void**)&gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for index LUT");
	if(cudaMemcpy(gpu_c_lut, bxf->c_lut, c_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy index LUT to GPU");
	if(cudaBindTexture(0, tex_c_lut, gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_c_lut to linear memory");
	total_bytes += c_lut_mem_size;

	// Allocate memory to hold the calculated dc_dv values.
	dc_dv_mem_size = num_tiles * 3 * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv stream on GPU");
	if(cudaBindTexture(0, tex_dc_dv, gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv to linear memory");
	total_bytes += dc_dv_mem_size;

	// Allocate memory to hold the calculated score values.
	score_mem_size = num_tiles * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the score stream on GPU");
	if(cudaBindTexture(0, tex_score, gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_score to linear memory");
	total_bytes += score_mem_size;

	// Allocate memory to hold the gradient values.
	if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad stream on GPU");
	if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
	if(cudaBindTexture(0, tex_grad, gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_grad to linear memory");
	total_bytes += 2 * coeff_mem_size;

	printf("DONE!\n");
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
}

/***********************************************************************
 * bspline_cuda_initialize_d
 *
 * Initialize the GPU to execute bspline_cuda_score_d_mse().
 ***********************************************************************/
void bspline_cuda_initialize_d(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	printf("Initializing CUDA... ");

	unsigned int total_bytes = 0;

	// Copy the fixed image to the GPU.
	if(cudaMalloc((void**)&gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for fixed image");
	if(cudaMemcpy(gpu_fixed_image, fixed->img, fixed->npix * fixed->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy fixed image to GPU");
	if(cudaBindTexture(0, tex_fixed_image, gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_fixed_image to linear memory");
	total_bytes += fixed->npix * fixed->pix_size;

	// Copy the moving image to the GPU.
	if(cudaMalloc((void**)&gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving image");
	if(cudaMemcpy(gpu_moving_image, moving->img, moving->npix * moving->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving image to GPU");
	if(cudaBindTexture(0, tex_moving_image, gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_image to linear memory");
	total_bytes += moving->npix * moving->pix_size;

	// Copy the moving gradient to the GPU.
	if(cudaMalloc((void**)&gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving gradient");
	if(cudaMemcpy(gpu_moving_grad, moving_grad->img, moving_grad->npix * moving_grad->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving gradient to GPU");
	if(cudaBindTexture(0, tex_moving_grad, gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_grad to linear memory");
	total_bytes += moving_grad->npix * moving_grad->pix_size;

	// Allocate memory for the coefficient LUT on the GPU.  The LUT will be copied to the
	// GPU each time bspline_cuda_score_d_mse is called.
	coeff_mem_size = sizeof(float) * bxf->num_coeff;
	if(cudaMalloc((void**)&gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for coefficient LUT");
	if(cudaBindTexture(0, tex_coeff, gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_coeff to linear memory");
	total_bytes += coeff_mem_size;

	// Copy the multiplier LUT to the GPU.
	size_t q_lut_mem_size = sizeof(float)
		* bxf->vox_per_rgn[0]
		* bxf->vox_per_rgn[1]
		* bxf->vox_per_rgn[2]
		* 64;
	if(cudaMalloc((void**)&gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for multiplier LUT");
	if(cudaMemcpy(gpu_q_lut, bxf->q_lut, q_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy multiplier LUT to GPU");
	if(cudaBindTexture(0, tex_q_lut, gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_q_lut to linear memory");
	total_bytes += q_lut_mem_size;

	// Copy the index LUT to the GPU.
	size_t c_lut_mem_size = sizeof(int) 
		* bxf->rdims[0] 
		* bxf->rdims[1] 
		* bxf->rdims[2] 
		* 64;
	if(cudaMalloc((void**)&gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for index LUT");
	if(cudaMemcpy(gpu_c_lut, bxf->c_lut, c_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy index LUT to GPU");
	if(cudaBindTexture(0, tex_c_lut, gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_c_lut to linear memory");
	total_bytes += c_lut_mem_size;

	// Allocate memory to hold the calculated dc_dv values.
	dc_dv_mem_size = 3 * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv stream on GPU");
	if(cudaBindTexture(0, tex_dc_dv, gpu_dc_dv, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dc_dv to linear memory");
	total_bytes += dc_dv_mem_size;

	// Allocate memory to hold the calculated score values.
	score_mem_size = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2] * sizeof(float);
	if(cudaMalloc((void**)&gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the score stream on GPU");
	if(cudaBindTexture(0, tex_score, gpu_score, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_score to linear memory");
	total_bytes += score_mem_size;

	// Allocate memory to hold the gradient values.
	if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad stream on GPU");
	if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
	if(cudaBindTexture(0, tex_grad, gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_grad to linear memory");
	total_bytes += 2 * coeff_mem_size;

	printf("DONE!\n");
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
}

/***********************************************************************
 * bspline_cuda_initialize
 ***********************************************************************/
void bspline_cuda_initialize(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	printf("Initializing CUDA... ");

	unsigned int total_bytes = 0;

	// Copy the fixed image to the GPU.
	if(cudaMalloc((void**)&gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for fixed image");
	if(cudaMemcpy(gpu_fixed_image, fixed->img, fixed->npix * fixed->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy fixed image to GPU");
	if(cudaBindTexture(0, tex_fixed_image, gpu_fixed_image, fixed->npix * fixed->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_fixed_image to linear memory");
	total_bytes += fixed->npix * fixed->pix_size;

	// Copy the moving image to the GPU.
	if(cudaMalloc((void**)&gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving image");
	if(cudaMemcpy(gpu_moving_image, moving->img, moving->npix * moving->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving image to GPU");
	if(cudaBindTexture(0, tex_moving_image, gpu_moving_image, moving->npix * moving->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_image to linear memory");
	total_bytes += moving->npix * moving->pix_size;

	// Copy the moving gradient to the GPU.
	if(cudaMalloc((void**)&gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for moving gradient");
	if(cudaMemcpy(gpu_moving_grad, moving_grad->img, moving_grad->npix * moving_grad->pix_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy moving gradient to GPU");
	if(cudaBindTexture(0, tex_moving_grad, gpu_moving_grad, moving_grad->npix * moving_grad->pix_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_moving_grad to linear memory");
	total_bytes += moving_grad->npix * moving_grad->pix_size;

	// Allocate memory for the coefficient LUT on the GPU.  The LUT will be copied to the
	// GPU each time bspline_cuda_run_kernels is called.
	coeff_mem_size = sizeof(float) * bxf->num_coeff;
	if(cudaMalloc((void**)&gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for coefficient LUT");
	if(cudaBindTexture(0, tex_coeff, gpu_coeff, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_coeff to linear memory");
	total_bytes += coeff_mem_size;

	// Copy the multiplier LUT to the GPU.
	size_t q_lut_mem_size = sizeof(float)
		* bxf->vox_per_rgn[0]
		* bxf->vox_per_rgn[1]
		* bxf->vox_per_rgn[2]
		* 64;
	if(cudaMalloc((void**)&gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for multiplier LUT");
	if(cudaMemcpy(gpu_q_lut, bxf->q_lut, q_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy multiplier LUT to GPU");
	if(cudaBindTexture(0, tex_q_lut, gpu_q_lut, q_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_q_lut to linear memory");
	total_bytes += q_lut_mem_size;

	// Copy the index LUT to the GPU.
	size_t c_lut_mem_size = sizeof(int) 
		* bxf->rdims[0] 
		* bxf->rdims[1] 
		* bxf->rdims[2] 
		* 64;
	if(cudaMalloc((void**)&gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for index LUT");
	if(cudaMemcpy(gpu_c_lut, bxf->c_lut, c_lut_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy index LUT to GPU");
	if(cudaBindTexture(0, tex_c_lut, gpu_c_lut, c_lut_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_c_lut to linear memory");
	total_bytes += c_lut_mem_size;

	// Allocate memory to hold the voxel displacement values.
	size_t volume_mem_size = fixed->npix * fixed->pix_size;
	if(cudaMalloc((void**)&gpu_dx, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for dy stream on GPU");
	if(cudaMalloc((void**)&gpu_dy, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for dx stream on GPU");
	if(cudaMalloc((void**)&gpu_dz, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for dz stream on GPU");
	total_bytes += volume_mem_size * 3;

	if(cudaBindTexture(0, tex_dx, gpu_dx, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dx to linear memory");
	if(cudaBindTexture(0, tex_dy, gpu_dy, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dy to linear memory");
	if(cudaBindTexture(0, tex_dz, gpu_dz, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_dz to linear memory");

	// Allocate memory to hold the calculated intensity difference values.
	if(cudaMalloc((void**)&gpu_diff, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the diff stream on GPU");
	total_bytes += volume_mem_size;

	// Allocate memory to hold the array of valid voxels;
	if(cudaMalloc((void**)&gpu_valid_voxels, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the valid_voxel stream on GPU");
	total_bytes += volume_mem_size;

	// Allocate memory to hold the calculated dc_dv values.
	if(cudaMalloc((void**)&gpu_dc_dv_x, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv_x stream on GPU");
	if(cudaMalloc((void**)&gpu_dc_dv_y, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv_x stream on GPU");
	if(cudaMalloc((void**)&gpu_dc_dv_z, volume_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the dc_dv_x stream on GPU");
	total_bytes += 3 * volume_mem_size;

	// Allocate memory to hold the gradient values.
	if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad stream on GPU");
	if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
	total_bytes += 2 * coeff_mem_size;

	printf("DONE!\n");
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
}

/***********************************************************************
 * bspline_cuda_copy_coeff_lut
 *
 * This function copies the coefficient LUT to the GPU in preparation
 * for calculating the score.
 ***********************************************************************/
void bspline_cuda_copy_coeff_lut(
	BSPLINE_Xform *bxf)
{
	// Copy the coefficient LUT to the GPU.
	if(cudaMemcpy(gpu_coeff, bxf->coeff, coeff_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy coefficient LUT to GPU");
}

/***********************************************************************
 * bspline_cuda_clear_score
 *
 * This function sets all the elements in the score stream to 0.
 ***********************************************************************/
void bspline_cuda_clear_score() 
{
	if(cudaMemset(gpu_score, 0, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to clear the score stream on GPU\n");
}

/***********************************************************************
 * bspline_cuda_clear_grad
 *
 * This function sets all the elements in the gradient stream to 0.
 ***********************************************************************/
void bspline_cuda_clear_grad() 
{
	if(cudaMemset(gpu_grad, 0, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to clear the grad stream on GPU\n");
}

/***********************************************************************
 * bspline_cuda_clear_dc_dv
 *
 * This function sets all the elements in the dc_dv stream to 0.
 ***********************************************************************/
void bspline_cuda_clear_dc_dv() 
{
	if(cudaMemset(gpu_dc_dv, 0, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("Failed to clear the dc_dv stream on GPU\n");
}

/***********************************************************************
 * bspline_cuda_copy_grad_to_host
 *
 * This function copies the gradient stream to the host.
 ***********************************************************************/
void bspline_cuda_copy_grad_to_host (float* host_grad)
{
    if (cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");
}

/***********************************************************************
 * bspline_cuda_calculate_run_kernels_g
 *
 * This function runs the kernels to calculate the score and gradient
 * as part of bspline_cuda_score_g_mse.
 ***********************************************************************/
void bspline_cuda_calculate_run_kernels_g(
					  Volume *fixed,
					  Volume *moving,
					  Volume *moving_grad,
					  BSPLINE_Xform *bxf,
					  BSPLINE_Parms *parms,
					  int run_low_mem_version, 
					  int debug)
{
    FILE *fp;
    char debug_fn[1024];

#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
#endif
	
    // Dimensions of the volume (in tiles)
    int3 rdims;			
    rdims.x = bxf->rdims[0];
    rdims.y = bxf->rdims[1];
    rdims.z = bxf->rdims[2];

    // Number of knots
    int3 cdims;
    cdims.x = bxf->cdims[0];
    cdims.y = bxf->cdims[1];
    cdims.z = bxf->cdims[2];

    // Dimensions of the volume (in voxels)
    int3 volume_dim;		
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Image origin (in mm)
    float3 img_origin;		
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Image spacing (in mm)
    float3 img_spacing;     
    img_spacing.x = (float)bxf->img_spacing[0];
    img_spacing.y = (float)bxf->img_spacing[1];
    img_spacing.z = (float)bxf->img_spacing[2];

    // Image offset
    float3 img_offset;     
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Pixel spacing
    float3 pix_spacing;     
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    // Configure the grid.
    int threads_per_block;
    int num_threads;
    int num_blocks;
    int smemSize;

#if defined (commentout)
    if (debug) {
	sprintf (debug_fn, "dump_mse.txt");
	fp = fopen (debug_fn, "w");
    }
#endif

    if (!run_low_mem_version) {
	//	printf("Launching one-shot version of bspline_cuda_score_g_mse_kernel1...\n");
		
	threads_per_block = 256;
	num_threads = fixed->npix;
	num_blocks = (int)ceil(num_threads / (float)threads_per_block);
	dim3 dimGrid1(num_blocks / 128, 128, 1);
	dim3 dimBlock1(threads_per_block, 1, 1);
	smemSize = 12 * sizeof(float) * threads_per_block;

	bspline_cuda_score_g_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>
		(
		 gpu_dc_dv,
		 gpu_score,
		 gpu_coeff,
		 gpu_fixed_image,
		 gpu_moving_image,
		 gpu_moving_grad,
		 volume_dim,
		 img_origin,
		 img_spacing,
		 img_offset,
		 roi_offset,
		 roi_dim,
		 vox_per_rgn,
		 pix_spacing,
		 rdims,
		 cdims);

#if defined (commentout)
	if (debug) {
	    int ri, rj, rk;
	    int fi, fj, fk;
	    float *tmp = (float*) malloc (dc_dv_mem_size);
	    if (cudaMemcpy (tmp, gpu_dc_dv, dc_dv_mem_size,
			    cudaMemcpyDeviceToHost) != cudaSuccess) {
		checkCUDAError("Failed to copy gpu_dc_dv to CPU");
	    }

	    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
		for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
		    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
			int idx = 3 * (((rk * bxf->roi_dim[1]) + rj) * bxf->roi_dim[0] + ri);
			fprintf (fp, "%d %d %d %g %g %g\n", ri, rj, rk, 
				 tmp[idx+0], tmp[idx+1], tmp[idx+2]);
		    }
		}
	    }
	    free (tmp);
	}
#endif

    } else {
	int tiles_per_launch = 512;
	printf("Launching low memory version of bspline_cuda_score_g_mse_kernel1 with %d tiles per launch. \n", tiles_per_launch);
		
	threads_per_block = 256;
	num_threads = tiles_per_launch * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
	num_blocks = (int)ceil(num_threads / (float)threads_per_block);
	dim3 dimGrid1(num_blocks / 128, 128, 1);
	dim3 dimBlock1(threads_per_block, 1, 1);
	smemSize = 12 * sizeof(float) * threads_per_block;

	for (int i = 0; i < rdims.x * rdims.y * rdims.z; i += tiles_per_launch) {
	    bspline_cuda_score_g_mse_kernel1_low_mem<<<dimGrid1, dimBlock1, smemSize>>>
		    (
		     gpu_dc_dv,
		     gpu_score,
		     i,
		     tiles_per_launch,
		     volume_dim,
		     img_origin,
		     img_spacing,
		     img_offset,
		     roi_offset,
		     roi_dim,
		     vox_per_rgn,
		     pix_spacing,
		     rdims,
		     cdims);
	}

    }

#if defined (commentout)
    if (debug) {
	fclose (fp);
    }
#endif


    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_g_mse_compute_score failed");

#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("%f seconds to run bspline_cuda_score_g_mse_kernel1 \n", 
	   double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    // Reconfigure the grid.
    threads_per_block = 256;
    num_threads = bxf->num_knots;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
    smemSize = 15 * sizeof(float) * threads_per_block;

    //printf("Launching bspline_cuda_score_f_mse_kernel2...");
    bspline_cuda_score_g_mse_kernel2<<<dimGrid2, dimBlock2, smemSize>>>
	    (
	     gpu_dc_dv,
	     gpu_grad,
	     num_threads,
	     rdims,
	     cdims,
	     vox_per_rgn);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_g_mse_kernel2 failed");

#if defined (commentout)
    if (1) {
	float *host_grad = (float*) malloc (coeff_mem_size);
	if (cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, 
		       cudaMemcpyDeviceToHost) != cudaSuccess) {
	    checkCUDAError("Failed to copy gpu_grad to CPU");
	}

	/* kkk */
	printf ("host_grad[0] = %g\n", host_grad[0]);
	printf ("host_grad[5] = %g\n", host_grad[5]);

	free (host_grad);
	exit (0);
    }
#endif

    //    printf ("bspline_cuda_score_g_mse_kernel2 complete.\n");

#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("%f seconds to run bspline_cuda_score_g_mse_kernel2\n", 
	   double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif
}

/***********************************************************************
 * bspline_cuda_calculate_run_kernels_f
 *
 * This function runs the kernels to calculate the score and gradient
 * as part of bspline_cuda_score_f_mse.
 ***********************************************************************/
void
bspline_cuda_calculate_run_kernels_f
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms)
{
#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
#endif
	
    // Dimensions of the volume (in tiles)
    int3 rdims;			
    rdims.x = bxf->rdims[0];
    rdims.y = bxf->rdims[1];
    rdims.z = bxf->rdims[2];

    // Number of knots
    int3 cdims;
    cdims.x = bxf->cdims[0];
    cdims.y = bxf->cdims[1];
    cdims.z = bxf->cdims[2];

    // Dimensions of the volume (in voxels)
    int3 volume_dim;		
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Image origin (in mm)
    float3 img_origin;		
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Image spacing (in mm)
    float3 img_spacing;     
    img_spacing.x = (float)bxf->img_spacing[0];
    img_spacing.y = (float)bxf->img_spacing[1];
    img_spacing.z = (float)bxf->img_spacing[2];

    // Image offset
    float3 img_offset;     
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Pixel spacing
    float3 pix_spacing;     
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif
    /*
    // Configure the grid.
    int threads_per_block = 256;
    int num_threads = fixed->npix;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid1(num_blocks / 128, 128, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);

    // printf("Launching bspline_cuda_score_f_mse_kernel1...\n");
	
    bspline_cuda_score_f_mse_kernel1<<<dimGrid1, dimBlock1>>>(
    gpu_dc_dv,
    gpu_score,
    gpu_c_lut,
    gpu_q_lut,
    gpu_coeff,
    gpu_fixed_image,
    gpu_moving_image,
    gpu_moving_grad,
    volume_dim,
    img_origin,
    img_spacing,
    img_offset,
    roi_offset,
    roi_dim,
    vox_per_rgn,
    pix_spacing,
    rdims);
	
    if(cudaThreadSynchronize() != cudaSuccess)
    checkCUDAError("\bspline_cuda_score_f_mse_kernel1 failed");
    */

	
    // Configure the grid.
    int threads_per_block = 256;
    int num_threads = fixed->npix;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid11(num_blocks, 1, 1);
    dim3 dimBlock11(threads_per_block, 1, 1);

    // printf("Launching bspline_cuda_score_f_mse_kernel1...\n");
    bspline_cuda_score_f_mse_compute_score<<<dimGrid11, dimBlock11>>>(
								      gpu_dc_dv,
								      gpu_score,
								      gpu_diff,
								      gpu_mvr,
								      volume_dim,
								      img_origin,
								      img_spacing,
								      img_offset,
								      roi_offset,
								      roi_dim,
								      vox_per_rgn,
								      pix_spacing,
								      rdims);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_f_mse_compute_score failed");

    threads_per_block = 256;
    num_threads = 3 * fixed->npix;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid12((int)ceil(num_blocks / 64.0), 64, 1);
    dim3 dimBlock12(threads_per_block, 1, 1);

    bspline_cuda_score_f_compute_dc_dv<<<dimGrid12, dimBlock12>>>(
								  gpu_dc_dv,
								  volume_dim,
								  vox_per_rgn,
								  roi_offset,
								  roi_dim,
								  rdims);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_f_compute_dc_dv failed");
	
#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("%f seconds to run bspline_cuda_score_f_mse_kernel1\n", 
	   double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    // Reconfigure the grid.
    threads_per_block = 256;
    num_threads = bxf->num_knots;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
    int smemSize = 15 * sizeof(float) * threads_per_block;

    //printf("Launching bspline_cuda_score_f_mse_kernel2...");
	
    bspline_cuda_score_f_mse_kernel2_nk<<<dimGrid2, dimBlock2, smemSize>>>(
									   gpu_dc_dv,
									   gpu_grad,
									   num_threads,
									   rdims,
									   cdims,
									   vox_per_rgn);

    /*
      bspline_cuda_score_f_mse_kernel2_v2<<<dimGrid2, dimBlock2, smemSize>>>(
      gpu_grad,
      num_threads,
      rdims,
      cdims,
      vox_per_rgn);
    */
	

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_f_mse_kernel2 failed");

#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("%f seconds to run bspline_cuda_score_f_mse_kernel2\n", 
	   double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif
}

/***********************************************************************
 * bspline_cuda_final_steps_f
 *
 * This function performs sum reduction of the score and gradient
 * streams as part of bspline_cuda_score_f_mse.
 ***********************************************************************/
void 
bspline_cuda_final_steps_f
(
 BSPLINE_Parms* parms, 
 BSPLINE_Xform* bxf,
 Volume *fixed,
 int   *vox_per_rgn,
 int   *volume_dim,
 float *host_score,
 float *host_grad,
 float *host_grad_mean,
 float *host_grad_norm)
{
    //int num_elems = vox_per_rgn[0] * vox_per_rgn[1] * vox_per_rgn[2];
    int num_elems = volume_dim[0] * volume_dim[1] * volume_dim[2];
    int num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(128, 2, 2);
    int smemSize = 512 * sizeof(float);
	
    // Calculate the score.
    // printf("Launching sum_reduction_kernel... ");
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
							  gpu_score,
							  gpu_score,
							  num_elems
							  );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_score_kernel failed");

    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							  gpu_score,
							  gpu_score,
							  num_elems
							  );
	
    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("sum_reduction_last_step_kernel failed");

    if(cudaMemcpy(host_score, gpu_score,  sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy score from GPU to host");

    *host_score = *host_score / (volume_dim[0] * volume_dim[1] * volume_dim[2]);

    FILE *scores_file; 
    scores_file = fopen("scores.txt", "a+");
    fprintf(scores_file, "%f\n", *host_score);
    fclose(scores_file);

    // Calculate grad_norm and grad_mean.

    // Reconfigure the grid.
    int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
    num_elems = bxf->num_coeff;
    num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(128, 2, 2);
    smemSize = 512 * sizeof(float);

    //    printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
							     gpu_grad,
							     num_vox,
							     num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");

    if(cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");

#if defined (commentout)
    /* kkk */
    printf ("host_grad[0] = %g\n", host_grad[0]);
    printf ("host_grad[5] = %g\n", host_grad[5]);
    exit (0);
#endif

    // printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid2, dimBlock2, smemSize>>>
	    (
	     gpu_grad,
	     gpu_grad_temp,
	     num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_mean_kernel failed");

    if(cudaMemcpy(host_grad_mean, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_mean from GPU to host");

    // printf("Launching bspline_cuda_compute_grad_norm_kernel... ");
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize>>>
	    (
	     gpu_grad,
	     gpu_grad_temp,
	     num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_norm_kernel failed");

    if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_norm from GPU to host");
}

/***********************************************************************
 * bspline_cuda_calculate_score_e
 *
 * This function runs the kernel to compute the score values for the
 * entire volume as part of bspline_cuda_score_e_mse.
 ***********************************************************************/
void bspline_cuda_calculate_score_e
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms)
{
    // Dimensions of the volume (in tiles)
    float3 rdims;			
    rdims.x = (float)bxf->rdims[0];
    rdims.y = (float)bxf->rdims[1];
    rdims.z = (float)bxf->rdims[2];

    // Dimensions of the volume (in voxels)
    int3 volume_dim;		
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Image origin (in mm)
    float3 img_origin;		
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Image spacing (in mm)
    float3 img_spacing;     
    img_spacing.x = (float)bxf->img_spacing[0];
    img_spacing.y = (float)bxf->img_spacing[1];
    img_spacing.z = (float)bxf->img_spacing[2];

    // Image offset
    float3 img_offset;     
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Pixel spacing
    float3 pix_spacing;     
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];
	
    // Configure the grid.
    int threads_per_block = 256;
    int num_threads = volume_dim.x * volume_dim.y * volume_dim.z;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(threads_per_block, 1, 1);

    // printf("Launching bspline_cuda_score_e_mse_kernel1a... ");
    bspline_cuda_score_e_mse_kernel1a<<<dimGrid, dimBlock>>>(
							     gpu_dc_dv,
							     gpu_score,
							     rdims,
							     volume_dim,
							     img_origin,
							     img_spacing,
							     img_offset,
							     roi_offset,
							     roi_dim,
							     vox_per_rgn,
							     pix_spacing
							     );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_score_e_mse_kernel1a failed");
}

/***********************************************************************
 * bspline_cuda_run_kernels_e_v2
 *
 * This function runs the kernel to compute the dc_dv values for a given
 * set as part of bspline_cuda_score_e_mse.  The calculation of the score
 * values is handled by bspline_cuda_calculate_score_e.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_e_v2
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms,
 int sidx0,
 int sidx1,
 int sidx2)
{
    //LARGE_INTEGER clock_count, clock_frequency;
    //double clock_start, clock_end;
    //QueryPerformanceFrequency(&clock_frequency);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;
	
    // Dimensions of the volume (in tiles)
    float3 rdims;			
    rdims.x = (float)bxf->rdims[0];
    rdims.y = (float)bxf->rdims[1];
    rdims.z = (float)bxf->rdims[2];

    // Dimensions of the set (in tiles)
    int3 sdims;				
    sdims.x = (int)ceil(rdims.x / 4.0);
    sdims.y = (int)ceil(rdims.y / 4.0);
    sdims.z = (int)ceil(rdims.z / 4.0);

    // Dimensions of the volume (in voxels)
    int3 volume_dim;		
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Image origin (in mm)
    float3 img_origin;		
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Image spacing (in mm)
    float3 img_spacing;     
    img_spacing.x = (float)bxf->img_spacing[0];
    img_spacing.y = (float)bxf->img_spacing[1];
    img_spacing.z = (float)bxf->img_spacing[2];

    // Image offset
    float3 img_offset;     
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Pixel spacing
    float3 pix_spacing;     
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

    int3 sidx;
    sidx.x = sidx0;
    sidx.y = sidx1;
    sidx.z = sidx2;
	
    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to read in the variables\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;

    // Clear the dc_dv values.
    bspline_cuda_clear_dc_dv();

    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to clear dc_dv values\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;

    // Run kernel #1.
    int threads_per_block = 256;
    int total_vox_per_rgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
    int num_tiles_per_set = sdims.x * sdims.y * sdims.z;
    int num_threads = total_vox_per_rgn * num_tiles_per_set;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid1(num_blocks, 1, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);

    // printf("Launching bspline_cuda_score_e_mse_kernel1b... ");
    bspline_cuda_score_e_mse_kernel1b<<<dimGrid1, dimBlock1>>>(
							       gpu_dc_dv,
							       gpu_score,
							       sidx,
							       rdims,
							       sdims,
							       volume_dim,
							       img_origin,
							       img_spacing,
							       img_offset,
							       roi_offset,
							       roi_dim,
							       vox_per_rgn,
							       total_vox_per_rgn,
							       pix_spacing
							       );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_score_e_mse_kernel1b failed");

    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to configure and run kernel1\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;

    /* The following code calculates the gradient by iterating through each tile
     * in the set.  The code following this section calculates the gradient for
     * the entire set at once, which improves parallelism and performance.

     // Reconfigure the grid.
     threads_per_block = 16;
     num_threads = 192;
     num_blocks = (int)ceil(num_threads / (float)threads_per_block);
     dim3 dimGrid2(num_blocks, 1, 1);
     dim3 dimBlock2(threads_per_block, 1, 1);

     // Update the control knots for each of the tiles in the set.
     int3 p;
     int3 s;
     int offset = 0;
     for(s.z = 0; s.z < sdims.z; s.z++) {
     for(s.y = 0; s.y < sdims.y; s.y++) {
     for(s.x = 0; s.x < sdims.x; s.x++) {

     p.x = (s.x * 4) + sidx.x;
     p.y = (s.y * 4) + sidx.y;
     p.z = (s.z * 4) + sidx.z;

     // printf("Launching bspline_cuda_score_d_mse_kernel2 for tile (%d, %d, %d)...\n", p.x, p.y, p.z);
     bspline_cuda_score_e_mse_kernel2_by_tiles<<<dimGrid2, dimBlock2>>>(
     gpu_dc_dv,
     gpu_grad,
     gpu_q_lut,
     num_threads,
     p,
     rdims,
     offset,
     vox_per_rgn,
     total_vox_per_rgn
     );

     if(cudaThreadSynchronize() != cudaSuccess)
     checkCUDAError("\nbspline_cuda_score_e_mse_kernel2 failed");

     offset++;
     }
     }
     }
    */
	
    threads_per_block = 16;
    num_threads = 192 * num_tiles_per_set;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);

    bspline_cuda_score_e_mse_kernel2_by_sets<<<dimGrid2, dimBlock2>>>(
								      gpu_dc_dv,
								      gpu_grad,
								      gpu_q_lut,
								      sidx,
								      sdims,
								      rdims,
								      vox_per_rgn,
								      192,
								      num_threads);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_score_e_mse_kernel2_by_sets failed");

    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to configure and run kernel2 64 times\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_run_kernels_e
 *
 * This function runs the kernels to compute both the score and dc_dv
 * values for a given set as part of bspline_cuda_score_e_mse.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_e
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms,
 int sidx0,
 int sidx1,
 int sidx2)
{
#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
#endif

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;
	
    // Dimensions of the volume (in tiles)
    float3 rdims;			
    rdims.x = (float)bxf->rdims[0];
    rdims.y = (float)bxf->rdims[1];
    rdims.z = (float)bxf->rdims[2];

    // Dimensions of the set (in tiles)
    int3 sdims;				
    sdims.x = (int)ceil(rdims.x / 4.0);
    sdims.y = (int)ceil(rdims.y / 4.0);
    sdims.z = (int)ceil(rdims.z / 4.0);

    // Dimensions of the volume (in voxels)
    int3 volume_dim;		
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Image origin (in mm)
    float3 img_origin;		
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Image spacing (in mm)
    float3 img_spacing;     
    img_spacing.x = (float)bxf->img_spacing[0];
    img_spacing.y = (float)bxf->img_spacing[1];
    img_spacing.z = (float)bxf->img_spacing[2];

    // Image offset
    float3 img_offset;     
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Pixel spacing
    float3 pix_spacing;     
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

    int3 sidx;
    sidx.x = sidx0;
    sidx.y = sidx1;
    sidx.z = sidx2;
	
    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to read in the variables\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;

    // Clear the dc_dv values.
    bspline_cuda_clear_dc_dv();

    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to clear dc_dv values\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;

    // Run kernel #1.
    int threads_per_block = 256;
    int total_vox_per_rgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
    int num_tiles_per_set = sdims.x * sdims.y * sdims.z;
    int num_threads = total_vox_per_rgn * num_tiles_per_set;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid1(num_blocks, 1, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);

    // printf("Launching bspline_cuda_score_e_mse_kernel1... ");
    bspline_cuda_score_e_mse_kernel1<<<dimGrid1, dimBlock1>>>(
							      gpu_dc_dv,
							      gpu_score,
							      sidx,
							      rdims,
							      sdims,
							      volume_dim,
							      img_origin,
							      img_spacing,
							      img_offset,
							      roi_offset,
							      roi_dim,
							      vox_per_rgn,
							      total_vox_per_rgn,
							      pix_spacing
							      );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_score_e_mse_kernel1 failed");

    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to configure and run kernel1\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

    //QueryPerformanceCounter(&clock_count);
    //clock_start = (double)clock_count.QuadPart;

    // Reconfigure the grid.
    int threadsPerControlPoint = 2;
    threads_per_block = 32;
    num_threads = 192 * threadsPerControlPoint;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
    int  smemSize = threadsPerControlPoint * threads_per_block * sizeof(float);

    // Update the control knots for each of the tiles in the set.
    int3 p;
    int3 s;
    int offset = 0;
    for(s.z = 0; s.z < sdims.z; s.z++) {
	for(s.y = 0; s.y < sdims.y; s.y++) {
	    for(s.x = 0; s.x < sdims.x; s.x++) {

		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		/*
		// printf("Launching bspline_cuda_score_d_mse_kernel2 for tile (%d, %d, %d)...\n", p.x, p.y, p.z);
		bspline_cuda_score_e_mse_kernel2_by_tiles<<<dimGrid2, dimBlock2>>>(
		gpu_dc_dv,
		gpu_grad,
		gpu_q_lut,
		num_threads,
		p,
		rdims,
		offset,
		vox_per_rgn,
		total_vox_per_rgn
		);
		*/
	
		bspline_cuda_score_e_mse_kernel2_by_tiles_v2<<<dimGrid2, dimBlock2, smemSize>>>(
												gpu_dc_dv,
												gpu_grad,
												gpu_q_lut,
												num_threads,
												p,
												rdims,
												offset,
												vox_per_rgn,
												threadsPerControlPoint
												);

		if(cudaThreadSynchronize() != cudaSuccess)
		    checkCUDAError("\nbspline_cuda_score_e_mse_kernel2 failed");

		offset++;
	    }
	}
    }

    //QueryPerformanceCounter(&clock_count);
    //clock_end = (double)clock_count.QuadPart;
    //printf("%f seconds to configure and run kernel2 64 times\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_final_steps_e_v2
 *
 * This function runs the kernels necessary to reduce the score and
 * gradient streams to a single value as part of 
 * bspline_cuda_score_e_mse_v2.  This version differs from 
 * bspline_cuda_score_e_mse in that the number of threads necessary to
 * reduce the score stream is different.
 ***********************************************************************/
void 
bspline_cuda_final_steps_e_v2
(
 BSPLINE_Parms* parms, 
 BSPLINE_Xform* bxf,
 Volume *fixed,
 int   *vox_per_rgn,
 int   *volume_dim,
 float *host_score,
 float *host_grad,
 float *host_grad_mean,
 float *host_grad_norm)
{
    // Start the clock.
    // LARGE_INTEGER clock_count, clock_frequency;
    // double clock_start, clock_end;
    // QueryPerformanceFrequency(&clock_frequency);
    // QueryPerformanceCounter(&clock_count);
    // clock_start = (double)clock_count.QuadPart;

    // Calculate the set dimensions.
#if defined (commentout)
    int3 sdims;
    sdims.x = (int)ceil(bxf->rdims[0] / 4.0);
    sdims.y = (int)ceil(bxf->rdims[1] / 4.0);
    sdims.z = (int)ceil(bxf->rdims[2] / 4.0);
#endif

    // Reduce the score stream to a single value.
    int threads_per_block = 512;
    int num_threads = volume_dim[0] * volume_dim[1] * volume_dim[2];
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(threads_per_block, 1, 1);
    int smemSize = threads_per_block * sizeof(float);

    // Calculate the score.
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
							  gpu_score,
							  gpu_score,
							  num_threads
							  );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_score_kernel failed");
	
    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							  gpu_score,
							  gpu_score,
							  num_threads
							  );
	
    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("sum_reduction_last_step_kernel failed");

    if(cudaMemcpy(host_score, gpu_score,  sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy score from GPU to host");

    *host_score = *host_score / (volume_dim[0] * volume_dim[1] * volume_dim[2]);

    // Calculate grad_norm and grad_mean.
    // Reconfigure the grid.
    int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
    int num_elems = bxf->num_coeff;
    num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(128, 2, 2);
    int smemSize2 = 512 * sizeof(float);

    // printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
							     gpu_grad,
							     num_vox,
							     num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");

    if(cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");

    // printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
									     gpu_grad,
									     gpu_grad_temp,
									     num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_mean_kernel failed");

    if(cudaMemcpy(host_grad_mean, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_mean from GPU to host");

    //printf("Launching bspline_cuda_compute_grad_norm_kernel... ");
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize2>>>(
									      gpu_grad,
									      gpu_grad_temp,
									      num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_norm_kernel failed");

    if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_norm from GPU to host");

    // Stop the clock.
    // QueryPerformanceCounter(&clock_count);
    // clock_end = (double)clock_count.QuadPart;
    // printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_final_steps_e
 *
 * This function runs the kernels necessary to reduce the score and
 * gradient streams to a single value as part of 
 * bspline_cuda_score_e_mse.  This version differs from 
 * bspline_cuda_score_e_mse_v2 in that the number of threads necessary to
 * reduce the score stream is different.
 ***********************************************************************/
void bspline_cuda_final_steps_e
(
 BSPLINE_Parms* parms, 
 BSPLINE_Xform* bxf,
 Volume *fixed,
 int   *vox_per_rgn,
 int   *volume_dim,
 float *host_score,
 float *host_grad,
 float *host_grad_mean,
 float *host_grad_norm)
{
#if defined (_WIN32)
    // Start the clock.
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    // Calculate the set dimensions.
    int3 sdims;
    sdims.x = (int)ceil(bxf->rdims[0] / 4.0);
    sdims.y = (int)ceil(bxf->rdims[1] / 4.0);
    sdims.z = (int)ceil(bxf->rdims[2] / 4.0);

    int threads_per_block = 512;
    int total_vox_per_rgn = vox_per_rgn[0] * vox_per_rgn[1] * vox_per_rgn[2];
    int num_tiles_per_set = sdims.x * sdims.y * sdims.z;
    int num_threads = total_vox_per_rgn * num_tiles_per_set;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(128, 2, 2);
    int smemSize = threads_per_block * sizeof(float);

    // Calculate the score.
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
							  gpu_score,
							  gpu_score,
							  num_threads
							  );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_score_kernel failed");
    else
	// printf("DONE!\n");

	sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							      gpu_score,
							      gpu_score,
							      num_threads
							      );
	
    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("sum_reduction_last_step_kernel failed");

    if(cudaMemcpy(host_score, gpu_score,  sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy score from GPU to host");

    *host_score = *host_score / (volume_dim[0] * volume_dim[1] * volume_dim[2]);

    // Calculate grad_norm and grad_mean.

    // Reconfigure the grid.
    int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
    int num_elems = bxf->num_coeff;
    num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(128, 2, 2);
    int smemSize2 = 512 * sizeof(float);

    // printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
							     gpu_grad,
							     num_vox,
							     num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");
    //else
    //	printf("DONE!\n");

    if(cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");

    // printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
									     gpu_grad,
									     gpu_grad_temp,
									     num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_mean_kernel failed");
    else
	// printf("DONE!\n");

	if(cudaMemcpy(host_grad_mean, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	    checkCUDAError("Failed to copy grad_mean from GPU to host");

    //printf("Launching bspline_cuda_compute_grad_norm_kernel... ");
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize2>>>(
									      gpu_grad,
									      gpu_grad_temp,
									      num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_norm_kernel failed");
    //else
    //	printf("DONE!\n");

    if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_norm from GPU to host");

    // Stop the clock.
#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    // printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif
}

/***********************************************************************
 * bspline_cuda_run_kernels_d
 *
 * This function runs the kernels to compute the score and dc_dv values
 * for a given tile as part of bspline_cuda_score_d_mse.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_d
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms,
 int p0,
 int p1,
 int p2)
{
    // Read in the dimensions of the volume.
    int3 volume_dim;
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Read in the dimensions of the region.
    float3 rdims;
    rdims.x = (float)bxf->rdims[0];
    rdims.y = (float)bxf->rdims[1];
    rdims.z = (float)bxf->rdims[2];

    // Read in spacing between the control knots.
    int3 vox_per_rgn;
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Read in the coordinates of the image origin.
    float3 img_origin;
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Read in the image spacing.
    float3 img_spacing;
    img_spacing.x = (float)bxf->img_spacing[0];
    img_spacing.y = (float)bxf->img_spacing[1];
    img_spacing.z = (float)bxf->img_spacing[2];

    // Read in image offset.
    float3 img_offset;
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Read in the voxel dimensions.
    float3 pix_spacing;
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    int3 roi_offset;
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    int3 roi_dim;
    roi_dim.x = bxf->roi_dim[0];
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

    // Read in the tile offset.
    int3 p;
    p.x = p0;
    p.y = p1;
    p.z = p2;

    // Start the clock.
#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    // Clear the dc_dv values.
    if(cudaMemset(gpu_dc_dv, 0, dc_dv_mem_size) != cudaSuccess)
	checkCUDAError("cudaMemset failed to fill gpu_dc_dv with 0\n");

    // printf("Launching bspline_cuda_score_d_mse_kernel1... ");

    /* KERNEL 1, VERSION 1 */
    int threads_per_block = 16;
    int num_threads = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
    int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(threads_per_block, 1, 1);

    bspline_cuda_score_d_mse_kernel1<<<dimGrid, dimBlock>>>(
							    gpu_dc_dv,
							    gpu_score,
							    p,
							    volume_dim,
							    img_origin,
							    img_spacing,
							    img_offset,
							    roi_offset,
							    roi_dim,
							    vox_per_rgn,
							    pix_spacing,
							    rdims
							    );

    /* KERNEL 1, VERSION 2
       int threads_per_block = 64;
       int num_threads = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
       int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
       dim3 dimGrid(num_blocks, 1, 1);
       dim3 dimBlock(threads_per_block, 1, 1);
	
       bspline_cuda_score_d_mse_kernel1_v2<<<dimGrid, dimBlock>>>(
       gpu_dc_dv,
       gpu_score,
       p,
       volume_dim,
       img_origin,
       img_spacing,
       img_offset,
       roi_offset,
       roi_dim,
       vox_per_rgn,
       pix_spacing,
       rdims
       );
    */

    /* KERNEL 1, VERSION 3
       int  threads_per_block = 128;
       int  threads_lost_per_block = threads_per_block - ((threads_per_block / 3) * 3);
       int  num_threads = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
       int  num_blocks = (int)ceil(num_threads / (float)(threads_per_block - threads_lost_per_block));
       dim3 dimGrid(num_blocks, 1, 1);
       dim3 dimBlock(threads_per_block, 1, 1);
       int  smemSize = 3 * ((threads_per_block - threads_lost_per_block) / 3) * sizeof(float);
       // printf("%d thread blocks will be created for each kernel.\n", num_blocks);
       // printf("smemSize = %d * sizeof(float)\n", 2 * ((threads_per_block - threads_lost_per_block) / 3));

       bspline_cuda_score_d_mse_kernel1_v3<<<dimGrid, dimBlock, smemSize>>>(
       gpu_dc_dv,
       gpu_score,
       p,
       volume_dim,
       img_origin,
       img_spacing,
       img_offset,
       roi_offset,
       roi_dim,
       vox_per_rgn,
       pix_spacing,
       rdims
       );
    */

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_score_d_mse_kernel1 failed");
    //else
    //printf("DONE!\n");

    /*
    // Reconfigure the grid.
    threads_per_block = 16;
    num_threads = 192;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
	
    // printf("Launching bspline_cuda_score_d_mse_kernel2... ");
    bspline_cuda_score_d_mse_kernel2<<<dimGrid2, dimBlock2>>>(
    gpu_dc_dv,
    gpu_grad,
    gpu_q_lut,
    num_threads,
    p,
    rdims,
    vox_per_rgn
    );
    */

    int threadsPerControlPoint = 1;
    threads_per_block = 32;
    num_threads = 192 * threadsPerControlPoint;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
    int  smemSize = threadsPerControlPoint * threads_per_block * sizeof(float);

    bspline_cuda_score_d_mse_kernel2_v2<<<dimGrid2, dimBlock2, smemSize>>>(
									   gpu_grad,
									   num_threads,
									   p,
									   rdims,
									   vox_per_rgn,
									   threadsPerControlPoint
									   );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_score_d_mse_kernel2 failed");
    //else
    //printf("DONE!\n");
	
    // Stop the clock.
#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    // printf("CUDA kernels for dc_dv and grad completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif
}

/***********************************************************************
 * bspline_cuda_final_steps_e
 *
 * This function runs the kernels necessary to reduce the score and
 * gradient streams to a single value as part of bspline_cuda_score_d_mse.
 ***********************************************************************/
void 
bspline_cuda_final_steps_d
(
 BSPLINE_Parms* parms, 
 BSPLINE_Xform* bxf,
 Volume *fixed,
 int   *vox_per_rgn,
 int   *volume_dim,
 float *host_score,
 float *host_grad,
 float *host_grad_mean,
 float *host_grad_norm)
{
    // Start the clock.
#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    int num_elems = vox_per_rgn[0] * vox_per_rgn[1] * vox_per_rgn[2];
    int num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(128, 2, 2);
    int smemSize = 512 * sizeof(float);
	
    // Calculate the score.
    // printf("Launching sum_reduction_kernel... ");
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
							  gpu_score,
							  gpu_score,
							  num_elems
							  );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_score_kernel failed");
    else
	// printf("DONE!\n");

	sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							      gpu_score,
							      gpu_score,
							      num_elems
							      );
	
    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("sum_reduction_last_step_kernel failed");

    if(cudaMemcpy(host_score, gpu_score,  sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy score from GPU to host");

    *host_score = *host_score / (volume_dim[0] * volume_dim[1] * volume_dim[2]);

    // Calculate grad_norm and grad_mean.

    // Reconfigure the grid.
    int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
    num_elems = bxf->num_coeff;
    num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(128, 2, 2);
    smemSize = 512 * sizeof(float);

    // printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
							     gpu_grad,
							     num_vox,
							     num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");
    else
	// printf("DONE!\n");

	if(cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	    checkCUDAError("Failed to copy gpu_grad to CPU");
		
    // printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
									     gpu_grad,
									     gpu_grad_temp,
									     num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_mean_kernel failed");
    else
	// printf("DONE!\n");

	if(cudaMemcpy(host_grad_mean, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	    checkCUDAError("Failed to copy grad_mean from GPU to host");

    // printf("Launching bspline_cuda_compute_grad_norm_kernel... ");
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
									     gpu_grad,
									     gpu_grad_temp,
									     num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
							    gpu_grad_temp,
							    gpu_grad_temp,
							    num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_norm_kernel failed");
    else
	// printf("DONE!\n");

	if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	    checkCUDAError("Failed to copy grad_norm from GPU to host");

    // Stop the clock.
#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("CUDA kernels for score completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif
}

/***********************************************************************
 * bspline_cuda_run_kernels_c
 *
 * This function runs the kernels necessary to compute the score and
 * dc_dv values as part of bspline_cuda_score_c_mse.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_c
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms,
 float *host_diff,
 float *host_dc_dv_x,
 float *host_dc_dv_y,
 float *host_dc_dv_z,
 float *host_score)
{
    // Read in the dimensions of the volume.
    int3 volume_dim;
    volume_dim.x = fixed->dim[0]; 
    volume_dim.y = fixed->dim[1];
    volume_dim.z = fixed->dim[2];

    // Read in the dimensions of the region.
    float3 rdims;
    rdims.x = (float)bxf->rdims[0];
    rdims.y = (float)bxf->rdims[1];
    rdims.z = (float)bxf->rdims[2];

    // Read in spacing between the control knots.
    int3 vox_per_rgn;
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // Read in the coordinates of the image origin.
    float3 img_origin;
    img_origin.x = (float)bxf->img_origin[0];
    img_origin.y = (float)bxf->img_origin[1];
    img_origin.z = (float)bxf->img_origin[2];

    // Read in image offset.
    float3 img_offset;
    img_offset.x = (float)moving->offset[0];
    img_offset.y = (float)moving->offset[1];
    img_offset.z = (float)moving->offset[2];

    // Read in the voxel dimensions.
    float3 pix_spacing;
    pix_spacing.x = (float)moving->pix_spacing[0];
    pix_spacing.y = (float)moving->pix_spacing[1];
    pix_spacing.z = (float)moving->pix_spacing[2];

    // Copy the coefficient LUT to the GPU.
    if(cudaMemcpy(gpu_coeff, bxf->coeff, coeff_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
	checkCUDAError("Failed to copy coefficient LUT to GPU");

    // Configure the grid.
    int num_elems = volume_dim.x * volume_dim.y * volume_dim.z;
    int num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(128, 2, 2);
    int smemSize = 512 * sizeof(float);
    printf("%d thread blocks will be created for each kernel.\n", num_blocks);

    // Start the clock.
#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    printf("Launching bspline_cuda_compute_dxyz_kernel... ");
    bspline_cuda_compute_dxyz_kernel<<<dimGrid, dimBlock>>>(
							    gpu_c_lut,
							    gpu_q_lut,
							    gpu_coeff,
							    volume_dim,
							    vox_per_rgn,
							    rdims,
							    gpu_dx,
							    gpu_dy,
							    gpu_dz
							    );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\nbspline_cuda_compute_dxyz_kernel failed");
    else
	printf("DONE!\n");

    printf("Launching bspline_cuda_compute_diff_kernel... ");
    bspline_cuda_compute_diff_kernel<<<dimGrid, dimBlock>>>(
							    gpu_fixed_image,
							    gpu_moving_image,
							    gpu_dx,
							    gpu_dy,
							    gpu_dz,
							    gpu_diff,
							    gpu_valid_voxels,
							    volume_dim,
							    img_origin,
							    pix_spacing,
							    img_offset
							    );
	
    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_diff_kernel failed");
    else
	printf("DONE!\n");
	
    if(cudaMemcpy(host_diff, gpu_diff, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy diff stream from GPU to host");

    printf("Launching bspline_cuda_compute_dc_dv_kernel... ");
    bspline_cuda_compute_dc_dv_kernel<<<dimGrid, dimBlock>>>(
							     gpu_fixed_image,
							     gpu_moving_image,
							     gpu_moving_grad,
							     gpu_c_lut, 
							     gpu_q_lut,
							     gpu_dx,
							     gpu_dy,
							     gpu_dz,
							     gpu_diff,
							     gpu_dc_dv_x,
							     gpu_dc_dv_y,
							     gpu_dc_dv_z,
							     // gpu_grad,
							     gpu_valid_voxels,
							     volume_dim,
							     vox_per_rgn,
							     rdims,
							     img_origin,
							     pix_spacing,
							     img_offset
							     );
	
    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_dc_dv_kernel failed");
    else
	printf("DONE!\n");

    printf("Launching bspline_cuda_compute_score_kernel... ");
    bspline_cuda_compute_score_kernel<<<dimGrid, dimBlock, smemSize>>>(
								       gpu_diff,
								       gpu_diff,
								       gpu_valid_voxels,
								       num_elems
								       );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_score_kernel failed");
    else
	printf("DONE!\n");

    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							  gpu_diff,
							  gpu_diff,
							  num_elems
							  );

    cudaThreadSynchronize();

    if(cudaMemcpy(host_score, gpu_diff,  sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy score from GPU to host");

    *host_score = *host_score / num_elems;

    // Stop the clock.
#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif

    // Copy results back from GPU.
    if(cudaMemcpy(host_dc_dv_x, gpu_dc_dv_x, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy dc_dv stream from GPU to host");
    if(cudaMemcpy(host_dc_dv_y, gpu_dc_dv_y, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy dc_dv stream from GPU to host");
    if(cudaMemcpy(host_dc_dv_z, gpu_dc_dv_z, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy dc_dv stream from GPU to host");

}

/***********************************************************************
 * bspline_cuda_calculate_gradient_c
 *
 * This function runs the kernels necessary to reduce the gradient
 * stream to a single value as part of bspline_cuda_score_c_mse.
 ***********************************************************************/
void 
bspline_cuda_calculate_gradient_c
(
 BSPLINE_Parms* parms, 
 Bspline_state* bst,
 BSPLINE_Xform* bxf,
 Volume *fixed,
 float *host_grad_norm,
 float *host_grad_mean) 
{
    BSPLINE_Score* ssd = &bst->ssd;
	
    // This copy is temporary until the gradient information is calculated on the GPU.
    // As soon as that is done, all the code in this function can be moved into the 
    // previous function.
    if(cudaMemcpy(gpu_grad, ssd->grad, coeff_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
	checkCUDAError("Failed to copy ssd->grad to GPU");

    // Configure the grid.
    int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
    int num_elems = bxf->num_coeff;
    int num_blocks = (int)ceil(num_elems / 512.0);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(128, 2, 2);
    int smemSize = 512 * sizeof(float);

    // Start the clock.
#if defined (_WIN32)
    LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
    QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
#endif

    printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid, dimBlock>>>(
							   gpu_grad,
							   num_vox,
							   num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");
    else
	printf("DONE!\n");

    if(cudaMemcpy(ssd->grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");

    printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid, dimBlock, smemSize>>>(
									   gpu_grad,
									   gpu_grad_temp,
									   num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							  gpu_grad_temp,
							  gpu_grad_temp,
							  num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_mean_kernel failed");
    else
	printf("DONE!\n");

    if(cudaMemcpy(host_grad_mean, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_mean from GPU to host");

    printf("Launching bspline_cuda_compute_grad_norm_kernel... ");
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid, dimBlock, smemSize>>>(
									   gpu_grad,
									   gpu_grad_temp,
									   num_elems);

    cudaThreadSynchronize();

    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
							  gpu_grad_temp,
							  gpu_grad_temp,
							  num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_grad_norm_kernel failed");
    else
	printf("DONE!\n");

    if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_norm from GPU to host");

    // Stop the clock.
#if defined (_WIN32)
    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
#endif
}

/***********************************************************************
 * bspline_cuda_clean_up_g
 *
 * This function frees all allocated memory on the GPU for version "g".
 ***********************************************************************/
void bspline_cuda_clean_up_g() 
{
    // Free memory on GPU.
    if(cudaFree(gpu_fixed_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for fixed_image");
    if(cudaFree(gpu_moving_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for moving_image");
    if(cudaFree(gpu_moving_grad) != cudaSuccess)
	checkCUDAError("Failed to free memory for moving_grad");
    if(cudaFree(gpu_coeff) != cudaSuccess) 
	checkCUDAError("Failed to free memory for coeff");
    if(cudaFree(gpu_dc_dv_x) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_x");
    if(cudaFree(gpu_dc_dv_y) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_y");
    if(cudaFree(gpu_dc_dv_z) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_z");
    if(cudaFree(gpu_score) != cudaSuccess)
	checkCUDAError("Failed to free memory for score");
}

/***********************************************************************
 * bspline_cuda_clean_up_f
 *
 * This function frees all allocated memory on the GPU for version "f".
 ***********************************************************************/
void bspline_cuda_clean_up_f() 
{
    // Free memory on GPU.
    if(cudaFree(gpu_fixed_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for fixed_image");
    if(cudaFree(gpu_moving_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for moving_image");
    if(cudaFree(gpu_moving_grad) != cudaSuccess)
	checkCUDAError("Failed to free memory for moving_grad");
    if(cudaFree(gpu_coeff) != cudaSuccess) 
	checkCUDAError("Failed to free memory for coeff");
    if(cudaFree(gpu_q_lut) != cudaSuccess) 
	checkCUDAError("Failed to free memory for q_lut");
    if(cudaFree(gpu_c_lut) != cudaSuccess) 
	checkCUDAError("Failed to free memory for c_lut");
    if(cudaFree(gpu_dc_dv_x) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_x");
    if(cudaFree(gpu_dc_dv_y) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_y");
    if(cudaFree(gpu_dc_dv_z) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_z");
    if(cudaFree(gpu_score) != cudaSuccess)
	checkCUDAError("Failed to free memory for score");

    if(cudaFree(gpu_diff) != cudaSuccess)
	checkCUDAError("Failed to free memory for diff");
    if(cudaFree(gpu_mvr) != cudaSuccess)
	checkCUDAError("Failed to free memory for mvr");

    printf("All memory on the GPU has been freed.\n");
}

/***********************************************************************
 * bspline_cuda_clean_up_d
 *
 * This function frees all allocated memory on the GPU for version "d"
 * and "e".
 ***********************************************************************/
void bspline_cuda_clean_up_d() 
{
    // Free memory on GPU.
    if(cudaFree(gpu_fixed_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for fixed_image");
    if(cudaFree(gpu_moving_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for moving_image");
    if(cudaFree(gpu_moving_grad) != cudaSuccess)
	checkCUDAError("Failed to free memory for moving_grad");
    if(cudaFree(gpu_coeff) != cudaSuccess) 
	checkCUDAError("Failed to free memory for coeff");
    if(cudaFree(gpu_q_lut) != cudaSuccess) 
	checkCUDAError("Failed to free memory for q_lut");
    if(cudaFree(gpu_c_lut) != cudaSuccess) 
	checkCUDAError("Failed to free memory for c_lut");
    if(cudaFree(gpu_dc_dv) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv");
    if(cudaFree(gpu_score) != cudaSuccess)
	checkCUDAError("Failed to free memory for score");

    printf("All memory on the GPU has been freed.\n");
}

/***********************************************************************
 * bspline_cuda_clean_up
 *
 * This function frees all allocated memory on the GPU for version "c".
 ***********************************************************************/
void bspline_cuda_clean_up() 
{
    // Free memory on GPU.
    if(cudaFree(gpu_fixed_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for fixed_image");
    if(cudaFree(gpu_moving_image) != cudaSuccess) 
	checkCUDAError("Failed to free memory for moving_image");
    if(cudaFree(gpu_moving_grad) != cudaSuccess)
	checkCUDAError("Failed to free memory for moving_grad");
    if(cudaFree(gpu_coeff) != cudaSuccess) 
	checkCUDAError("Failed to free memory for coeff");
    if(cudaFree(gpu_q_lut) != cudaSuccess) 
	checkCUDAError("Failed to free memory for q_lut");
    if(cudaFree(gpu_c_lut) != cudaSuccess) 
	checkCUDAError("Failed to free memory for c_lut");
    if(cudaFree(gpu_dx) != cudaSuccess)
	checkCUDAError("Failed to free memory for dx");
    if(cudaFree(gpu_dy) != cudaSuccess) 
	checkCUDAError("Failed to free memory for dy");
    if(cudaFree(gpu_dz) != cudaSuccess) 
	checkCUDAError("Failed to free memory for dz");
    if(cudaFree(gpu_diff) != cudaSuccess)
	checkCUDAError("Failed to free memory for diff");
    if(cudaFree(gpu_dc_dv_x) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_x");
    if(cudaFree(gpu_dc_dv_y) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_y");
    if(cudaFree(gpu_dc_dv_z) != cudaSuccess)
	checkCUDAError("Failed to free memory for dc_dv_z");
    if(cudaFree(gpu_valid_voxels) != cudaSuccess)
	checkCUDAError("Failed to free memory for valid_voxels");
    if(cudaFree(gpu_grad) != cudaSuccess)
	checkCUDAError("Failed to free memory for grad");
    if(cudaFree(gpu_grad_temp) != cudaSuccess)
	checkCUDAError("Failed to free memory for grad_temp");
}

/***********************************************************************
 * checkCUDAError
 *
 * If a CUDA error is detected, this function gets the error message
 * and displays it to the user.
 ***********************************************************************/
void checkCUDAError (const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) 
    {
	printf("CUDA Error -- %s: %s.\n", msg, cudaGetErrorString(err));
	exit(-1);
    } 
}
