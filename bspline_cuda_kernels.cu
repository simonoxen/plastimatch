#include "bspline_cuda.h"

/***********************************************************************
 * Declare texture references.
 ***********************************************************************/
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

/***********************************************************************
 * test_kernel
 *
 * A simple kernel used to ensure that CUDA is working correctly. 
 ***********************************************************************/
__global__ void test_kernel(
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
 ***********************************************************************/
__global__ void bspline_cuda_score_g_mse_kernel1 (
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

	float *A = &sdata[12*threadIdxInBlock + 0];
	float *B = &sdata[12*threadIdxInBlock + 4];
	float *C = &sdata[12*threadIdxInBlock + 8];
	float ii, jj, kk;
	float t1, t2, t3; 

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
			
			// Compute the q_lut values that pertain to this offset.
			ii = ((float)q.x) / vox_per_rgn.x;
			t3 = ii*ii*ii;
			t2 = ii*ii;
			t1 = ii;
			A[0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
			A[1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
			A[2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
			A[3] = (1.0/6.0) * (+ 1.0 * t3);

			jj = ((float)q.y) / vox_per_rgn.y;
			t3 = jj*jj*jj;
			t2 = jj*jj;
			t1 = jj;
			B[0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
			B[1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
			B[2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
			B[3] = (1.0/6.0) * (+ 1.0 * t3);

			kk = ((float)q.z) / vox_per_rgn.z;
			t3 = kk*kk*kk;
			t2 = kk*kk;
			t1 = kk;
			C[0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
			C[1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
			C[2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
			C[3] = (1.0/6.0) * (+ 1.0 * t3);

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			int k = 0;
			int3 t;
			for(t.z = 0; t.z < 4; t.z++) {
				for(t.y = 0; t.y < 4; t.y++) {
					for(t.x = 0; t.x < 4; t.x++) {

						// Calculate the index into the coefficients array.
						cidx = 3 * ((p.z + t.z) * cdims.x * cdims.y + (p.y + t.y) * cdims.x + (p.x + t.x));

						// Fetch the values for P, Ni, Nj, and Nk.
						P   = A[t.x] * B[t.y] * C[t.z];
						//N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
						//N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
						//N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value
						N.x = coeff[cidx+0];  // x-value
						N.y = coeff[cidx+1];  // y-value
						N.z = coeff[cidx+2];  // z-value

						// Update the output (v) values.
						d.x += P * N.x;
						d.y += P * N.y;
						d.z += P * N.z;

						k++;

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
				//m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				//m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				//m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				//m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				//m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				//m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				//m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				//m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_x1y1z1 = fx1 * fy1 * fz1 * moving_image[mvf];
				m_x2y1z1 = fx2 * fy1 * fz1 * moving_image[mvf + 1];
				m_x1y2z1 = fx1 * fy2 * fz1 * moving_image[mvf + volume_dim.x];
				m_x2y2z1 = fx2 * fy2 * fz1 * moving_image[mvf + volume_dim.x + 1];
				m_x1y1z2 = fx1 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x];
				m_x2y1z2 = fx2 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + 1];
				m_x1y2z2 = fx1 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x];
				m_x2y2z2 = fx2 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1];

				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				//diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				diff = fixed_image[fv] - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
				//dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				//dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				//dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
				dc_dv_element[0] = diff * moving_grad[3 * (int)mvr + 0];
				dc_dv_element[1] = diff * moving_grad[3 * (int)mvr + 1];
				dc_dv_element[2] = diff * moving_grad[3 * (int)mvr + 2];
			
			}
		}
	}
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
 ***********************************************************************/
__global__ void bspline_cuda_score_g_mse_kernel1_low_mem (
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
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_element = &dc_dv_element[3 * qidx];
			qidx = qidx * 64;
			
			// Compute the q_lut values that pertain to this offset.
			ii = ((float)q.x) / vox_per_rgn.x;
			t3 = ii*ii*ii;
			t2 = ii*ii;
			t1 = ii;
			A[0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
			A[1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
			A[2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
			A[3] = (1.0/6.0) * (+ 1.0 * t3);

			jj = ((float)q.y) / vox_per_rgn.y;
			t3 = jj*jj*jj;
			t2 = jj*jj;
			t1 = jj;
			B[0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
			B[1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
			B[2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
			B[3] = (1.0/6.0) * (+ 1.0 * t3);

			kk = ((float)q.z) / vox_per_rgn.z;
			t3 = kk*kk*kk;
			t2 = kk*kk;
			t1 = kk;
			C[0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
			C[1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
			C[2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
			C[3] = (1.0/6.0) * (+ 1.0 * t3);

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			int k = 0;
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

						k++;

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

__global__ void bspline_cuda_score_g_mse_kernel1_min_regs (
	float  *dc_dv,
	float  *score,			
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
	int    cidx;			// Index into the coefficient table

	float3 d;				// B-spline deformation vector
	float  diff;
	int3   i;

	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx2, fy2, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	
	float* dc_dv_element;

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (((gridDim.x * blockIdx.y) + blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z)) 
		+ ((blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x);

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
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z 
				* (((p.z * rdims.y + p.y) * rdims.x) + p.x)];

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			dc_dv_element = &dc_dv_element[3 * (((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x)];
			
			/*
			// Compute the q_lut values that pertain to this offset.
			diff = ((float)q.x) / vox_per_rgn.x;
			t.z = diff*diff*diff;
			t.y = diff*diff;
			t.x = diff;
			sdata[12*threadIdxInBlock + 0 + 0] = (1.0/6.0) * (- 1.0 * t.z + 3.0 * t.y - 3.0 * t.x + 1.0);
			sdata[12*threadIdxInBlock + 0 + 1] = (1.0/6.0) * (+ 3.0 * t.z - 6.0 * t.y             + 4.0);
			sdata[12*threadIdxInBlock + 0 + 2] = (1.0/6.0) * (- 3.0 * t.z + 3.0 * t.y + 3.0 * t.x + 1.0);
			sdata[12*threadIdxInBlock + 0 + 3] = (1.0/6.0) * (+ 1.0 * t.z);

			diff = ((float)q.y) / vox_per_rgn.y;
			t.z = diff*diff*diff;
			t.y = diff*diff;
			t.x = diff;
			sdata[12*threadIdxInBlock + 4 + 0] = (1.0/6.0) * (- 1.0 * t.z + 3.0 * t.y - 3.0 * t.x + 1.0);
			sdata[12*threadIdxInBlock + 4 + 1] = (1.0/6.0) * (+ 3.0 * t.z - 6.0 * t.y             + 4.0);
			sdata[12*threadIdxInBlock + 4 + 2] = (1.0/6.0) * (- 3.0 * t.z + 3.0 * t.y + 3.0 * t.x + 1.0);
			sdata[12*threadIdxInBlock + 4 + 3] = (1.0/6.0) * (+ 1.0 * t.z);

			diff = ((float)q.z) / vox_per_rgn.z;
			t.z = diff*diff*diff;
			t.y = diff*diff;
			t.x = diff;
			sdata[12*threadIdxInBlock + 8 + 0] = (1.0/6.0) * (- 1.0 * t.z + 3.0 * t.y - 3.0 * t.x + 1.0);
			sdata[12*threadIdxInBlock + 8 + 1] = (1.0/6.0) * (+ 3.0 * t.z - 6.0 * t.y             + 4.0);
			sdata[12*threadIdxInBlock + 8 + 2] = (1.0/6.0) * (- 3.0 * t.z + 3.0 * t.y + 3.0 * t.x + 1.0);
			sdata[12*threadIdxInBlock + 8 + 3] = (1.0/6.0) * (+ 1.0 * t.z);
			*/

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;
			
			for(i.z = 0; i.z < 4; i.z++) {
				for(i.y = 0; i.y < 4; i.y++) {
					for(i.x = 0; i.x < 4; i.x++) {

						// Calculate the index into the coefficients array.
						cidx = 3 * ((p.z + i.z) * cdims.x * cdims.y + (p.y + i.y) * cdims.x + (p.x + i.x));

						sdata[8*threadIdxInBlock + 3] = 1;

						// Compute the q_lut values that pertain to this offset.
						diff = ((float)q.x) / vox_per_rgn.x;
						sdata[8*threadIdxInBlock + 0] = diff*diff*diff;
						sdata[8*threadIdxInBlock + 1] = diff*diff;
						sdata[8*threadIdxInBlock + 2] = diff;
						switch(i.x) {
							case 0: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (- 1.0 * sdata[8*threadIdxInBlock + 0] + 3.0 * sdata[8*threadIdxInBlock + 1] - 3.0 * sdata[8*threadIdxInBlock + 2] + 1.0); break;
							case 1:	sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (+ 3.0 * sdata[8*threadIdxInBlock + 0] - 6.0 * sdata[8*threadIdxInBlock + 1]             + 4.0); break;
							case 2: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (- 3.0 * sdata[8*threadIdxInBlock + 0] + 3.0 * sdata[8*threadIdxInBlock + 1] + 3.0 * sdata[8*threadIdxInBlock + 2] + 1.0); break;
							case 3: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (+ 1.0 * sdata[8*threadIdxInBlock + 0]); break;
						}
						
						diff = ((float)q.y) / vox_per_rgn.y;
						sdata[8*threadIdxInBlock + 0] = diff*diff*diff;
						sdata[8*threadIdxInBlock + 1] = diff*diff;
						sdata[8*threadIdxInBlock + 2] = diff;
						switch(i.y) {
							case 0: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (- 1.0 * sdata[8*threadIdxInBlock + 0] + 3.0 * sdata[8*threadIdxInBlock + 1] - 3.0 * sdata[8*threadIdxInBlock + 2] + 1.0); break;
							case 1:	sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (+ 3.0 * sdata[8*threadIdxInBlock + 0] - 6.0 * sdata[8*threadIdxInBlock + 1]             + 4.0); break;
							case 2: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (- 3.0 * sdata[8*threadIdxInBlock + 0] + 3.0 * sdata[8*threadIdxInBlock + 1] + 3.0 * sdata[8*threadIdxInBlock + 2] + 1.0); break;
							case 3: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (+ 1.0 * sdata[8*threadIdxInBlock + 0]); break;
						}

						diff = ((float)q.z) / vox_per_rgn.z;
						sdata[8*threadIdxInBlock + 0] = diff*diff*diff;
						sdata[8*threadIdxInBlock + 1] = diff*diff;
						sdata[8*threadIdxInBlock + 2] = diff;
						switch(i.z) {
							case 0: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (- 1.0 * sdata[8*threadIdxInBlock + 0] + 3.0 * sdata[8*threadIdxInBlock + 1] - 3.0 * sdata[8*threadIdxInBlock + 2] + 1.0); break;
							case 1:	sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (+ 3.0 * sdata[8*threadIdxInBlock + 0] - 6.0 * sdata[8*threadIdxInBlock + 1]             + 4.0); break;
							case 2: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (- 3.0 * sdata[8*threadIdxInBlock + 0] + 3.0 * sdata[8*threadIdxInBlock + 1] + 3.0 * sdata[8*threadIdxInBlock + 2] + 1.0); break;
							case 3: sdata[8*threadIdxInBlock + 3] *= (1.0/6.0) * (+ 1.0 * sdata[8*threadIdxInBlock + 0]); break;
						}

						// Update the output values.
						d.x += sdata[8*threadIdxInBlock + 3] * tex1Dfetch(tex_coeff, cidx + 0);
						d.y += sdata[8*threadIdxInBlock + 3] * tex1Dfetch(tex_coeff, cidx + 1);
						d.z += sdata[8*threadIdxInBlock + 3] * tex1Dfetch(tex_coeff, cidx + 2);

					}
				}
			}

			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			displacement_in_vox.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			displacement_in_vox.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			displacement_in_vox.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_vox.x = displacement_in_vox.x + d.x;
			displacement_in_vox.y = displacement_in_vox.y + d.y;
			displacement_in_vox.z = displacement_in_vox.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_vox.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_vox.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_vox.z - img_offset.z) / pix_spacing.z;

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
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;	
				m_val = 0.0;
				m_val += (1.0 - fx2) * (1.0 - fy2) * (1.0 - fz2) * tex1Dfetch(tex_moving_image, mvf);
				m_val += fx2 * (1.0 - fy2) * (1.0 - fz2) * tex1Dfetch(tex_moving_image, mvf + 1);
				m_val += (1.0 - fx2) * fy2 * (1.0 - fz2) * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_val += fx2 * fy2 * (1.0 - fz2) * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_val += (1.0 - fx2) * (1.0 - fy2) * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_val += fx2 * (1.0 - fy2) * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_val += (1.0 - fx2) * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_val += fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);

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

__global__ void bspline_cuda_score_g_mse_kernel2 (
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
	float multiplier;

	//float3 temp0, temp1, temp2, temp3;

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

	float *temps = &sdata[15*threadIdxInBlock];
	temps[12] = 0.0;
	temps[13] = 0.0;
	temps[14] = 0.0;

	float ii, jj, kk;
	float t1, t2, t3;
	float A, B, C;
	int3 q;

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
						int3 t, t_temp;
						int c_lut_value;
						for(t_temp.z = 0; t_temp.z < 4; t_temp.z++) {
							for(t_temp.y = 0; t_temp.y < 4; t_temp.y++) {
								for(t_temp.x = 0; t_temp.x < 4; t_temp.x++) {
									c_lut_value = 
									    + (tileLocation.z + t_temp.z) * cdims.x * cdims.y
										+ (tileLocation.y + t_temp.y) * cdims.x 
										+ (tileLocation.x + t_temp.x);
									if(c_lut_value == threadIdxInGrid) {
										t.x = t_temp.x;
										t.y = t_temp.y;
										t.z = t_temp.z;
										break;
									}
								}
							}
						}

						// For all the voxels in this tile...
						for(q.z = 0, qidx = 0; q.z < vox_per_rgn.z; q.z++) {
							for(q.y = 0; q.y < vox_per_rgn.y; q.y++) {
								for(q.x = 0; q.x < vox_per_rgn.x; q.x++, qidx++) {

									// Compute the q_lut value.
									ii = (float)q.x / vox_per_rgn.x;
									t3 = ii*ii*ii;
									t2 = ii*ii;
									t1 = ii;
									switch(t.x) {
										case 0:
											A = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
											break;
										case 1:
											A = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
											break;
										case 2:
											A = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
											break;
										case 3:
											A = (1.0/6.0) * (+ 1.0 * t3);
											break;
									}

									jj = (float)q.y / vox_per_rgn.x;
									t3 = jj*jj*jj;
									t2 = jj*jj;
									t1 = jj;
									switch(t.y) {
										case 0:
											B = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
											break;
										case 1:
											B = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
											break;
										case 2:
											B = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
											break;
										case 3:
											B = (1.0/6.0) * (+ 1.0 * t3);
											break;
									}

									kk = (float)q.z / vox_per_rgn.x;
									t3 = kk*kk*kk;
									t2 = kk*kk;
									t1 = kk;
									switch(t.z) {
										case 0:
											C = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
											break;
										case 1:
											C = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
											break;
										case 2:
											C = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
											break;
										case 3:
											C = (1.0/6.0) * (+ 1.0 * t3);
											break;
									}

									multiplier = A * B * C;
									result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
									result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
									result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;
								}
							}
						}
					}
				}
			}
		}

		grad[3*threadIdxInGrid+0] = result.x;
		grad[3*threadIdxInGrid+1] = result.y;
		grad[3*threadIdxInGrid+2] = result.z;
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
__global__ void bspline_cuda_score_f_mse_compute_score (
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
				//cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				cidx = 3 * gpu_c_lut[pidx + k];

				// Fetch the values for P, Ni, Nj, and Nk.
				//P   = tex1Dfetch(tex_q_lut, qidx + k); 
				P   = gpu_q_lut[qidx + k];
				//N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				//N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				//N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				N.x = coeff[cidx+0];  // x-value
				N.y = coeff[cidx+1];  // y-value
				N.z = coeff[cidx+2];  // z-value

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
				//m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				//m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				//m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				//m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				//m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				//m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				//m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				//m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_x1y1z1 = fx1 * fy1 * fz1 * moving_image[mvf];
				m_x2y1z1 = fx2 * fy1 * fz1 * moving_image[mvf + 1];
				m_x1y2z1 = fx1 * fy2 * fz1 * moving_image[mvf + volume_dim.x];
				m_x2y2z1 = fx2 * fy2 * fz1 * moving_image[mvf + volume_dim.x + 1];
				m_x1y1z2 = fx1 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x];
				m_x2y1z2 = fx2 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + 1];
				m_x1y2z2 = fx1 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x];
				m_x2y2z2 = fx2 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1];
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				//diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				diff = fixed_image[fv] - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
				//dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				//dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				//dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
				dc_dv_element[0] = diff * moving_grad[3 * (int)mvr + 0];
				dc_dv_element[1] = diff * moving_grad[3 * (int)mvr + 1];
				dc_dv_element[2] = diff * moving_grad[3 * (int)mvr + 2];	
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