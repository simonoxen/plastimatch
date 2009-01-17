#include "bspline_cuda.h"

// Declare texture references.
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

texture<float, 1> tex_dc_dv;
texture<float, 1> tex_grad;

/* A simple kernel used to ensure that CUDA is working correctly. */
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
	float  fx, fy, fz;		// Physical coordinates within the volume.
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

			// Compute the physical coordinates of fixed image voxel.
			fx = img_origin.x + img_spacing.x * coord_in_volume.x;
			fy = img_origin.y + img_spacing.y * coord_in_volume.y;
			fz = img_origin.z + img_spacing.z * coord_in_volume.z;

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			// ----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			// ----------------------------------------------------------------

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
			
			// ----------------------------------------------------------------
			// Find correspondence in the moving image.
			// ----------------------------------------------------------------

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

				// ----------------------------------------------------------------
				// Compute interpolation fractions.
				// ----------------------------------------------------------------

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
				
				// ----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				// ----------------------------------------------------------------

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

				// ----------------------------------------------------------------
				// Compute intensity difference.
				// ----------------------------------------------------------------

				// diff[threadIdxInGrid] = fixed_image[threadIdxInGrid] - m_val;
				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				// ----------------------------------------------------------------
				// Accumulate the score.
				// ----------------------------------------------------------------

				score[threadIdxInGrid] += diff * diff;
				//score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				// ----------------------------------------------------------------
				// Compute dc_dv for this offset
				// ----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);

				//dc_dv[3*(qidx/64)+0] = (float)displacement_in_vox_round.x;
				//dc_dv[3*(qidx/64)+1] = (float)displacement_in_vox_round.y;
				//dc_dv[3*(qidx/64)+2] = (float)displacement_in_vox_round.z;
			}		
		}
	}
}

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
		float result = 0.0;

		int q[3];

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		
		// Calculate the linear index of the control point.
		m = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (m * 3);

		// Calculate index into coefficient texture.
		cidx = tex1Dfetch(tex_c_lut, 64*pidx + m) * 3;

		// Serial across offsets.
		for(int qidx = 0; qidx < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z); qidx++) {
			result += tex1Dfetch(tex_dc_dv, 3*qidx + offset) * tex1Dfetch(tex_q_lut, 64*qidx + m);
		}

		grad[cidx + offset] = tex1Dfetch(tex_grad, cidx + offset) + result;
	}
}

/* Kernel to compute the displacement values in the X, Y, and Z directions. */
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
	float* qrow;			// First element in the correct row in q_lut.
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

/* Kernel to compute the intensity difference between the voxels in the moving and fixed images. */
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

/* Kernel to compute the dc_dv values used to update the control-knot coefficients. */
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
	int3   p;		// Tile index.
	int3   q;		// Offset within tile.
	int    pidx;	// Index into c_lut.
	int    qidx;	// Index into q_lut.
	int    cidx;	// Index into the coefficient table.
	int*   prow;	// First element in the correct row in c_lut.
	float* qrow;	// First element in the correct row in q_lut.
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

// This reduce function will work for any size array.  It is the same as 
// bspline_cuda_compute_score_kernel, with the exception that it assumes all values are valid.
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

// This reduce function will work for any size array, and also checks a flag for each voxel
// to determine whether or not it is valid before adding it to the final sum.
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

	if(threadIdxInGrid < num_elems)
		grad[threadIdxInGrid] = 2 * grad[threadIdxInGrid] / num_vox;
}

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