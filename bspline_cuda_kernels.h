/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_cuda_kernels_h_
#define _bspline_cuda_kernels_h_

/* Function prototypes of kernels */
__global__ void kernel_deinterleave(
				int num_values,
				float* input,
				float* out_x,
				float* out_y,
				float* out_z);

__global__ void kernel_pad_64(
			float* input,
			float* output,
			int3 vol_dim,
			int3 tile_dim);

__global__ void kernel_pad(
			float* input,
			float* output,
			int3 vol_dim,
			int3 tile_dim);


__global__ void kernel_row_to_tile_major(
			float* input,
			float* output,
			int3 vol_dim,
			int3 tile_dim);


__global__ void kernel_bspline_mse_2_condense_64_texfetch(
				float* cond_x,		// Return: condensed dc_dv_x values
				float* cond_y,		// Return: condensed dc_dv_y values
				float* cond_z,		// Return: condensed dc_dv_z values
				float* dc_dv_x,		// Input : dc_dv_x values
				float* dc_dv_y,		// Input : dc_dv_y values
				float* dc_dv_z,		// Input : dc_dv_z values
				int* LUT_Tile_Offsets,	// Input : tile offsets
				int* LUT_Knot,		// Input : linear knot indicies
				int pad,		// Input : amount of tile padding
				int4 tile_dim,		// Input : dims of tiles
				float one_over_six);	// Input : Precomputed since GPU division is slow


__global__ void kernel_bspline_mse_2_condense_64(
				float* cond_x,		// Return: condensed dc_dv_x values
				float* cond_y,		// Return: condensed dc_dv_y values
				float* cond_z,		// Return: condensed dc_dv_z values
				float* dc_dv_x,		// Input : dc_dv_x values
				float* dc_dv_y,		// Input : dc_dv_y values
				float* dc_dv_z,		// Input : dc_dv_z values
				int* LUT_Tile_Offsets,	// Input : tile offsets
				int* LUT_Knot,		// Input : linear knot indicies
				int pad,		// Input : amount of tile padding
				int4 tile_dim,		// Input : dims of tiles
				float one_over_six);	// Input : Precomputed since GPU division is slow

__global__ void
kernel_bspline_mse_2_condense(
    float* cond_x,      // Return: condensed dc_dv_x values
    float* cond_y,      // Return: condensed dc_dv_y values
    float* cond_z,      // Return: condensed dc_dv_z values
    float* dc_dv_x,     // Input : dc_dv_x values
    float* dc_dv_y,     // Input : dc_dv_y values
    float* dc_dv_z,     // Input : dc_dv_z values
    int* LUT_Tile_Offsets,  // Input : tile offsets
    int* LUT_Knot,      // Input : linear knot indicies
    int pad,        // Input : amount of tile padding
    int4 tile_dim,      // Input : dims of tiles
    float one_over_six); // Input : Precomputed since GPU division is slow



__global__ void kernel_bspline_mse_2_reduce(
				float* grad,		// Return: interleaved dc_dp values
				float* cond_x,		// Input : condensed dc_dv_x values
				float* cond_y,		// Input : condensed dc_dv_y values
				float* cond_z);		// Input : condensed dc_dv_z values


__device__ float obtain_spline_basis_function(float one_over_six,
						  int t_idx, 
						  int vox_idx, 
						  int vox_per_rgn);



__global__ void
bspline_cuda_score_j_mse_kernel1 
(
    float  *dc_dv_x,       // OUTPUT
    float  *dc_dv_y,       // OUTPUT
    float  *dc_dv_z,       // OUTPUT
    float  *score,         // OUTPUT
    float  *coeff,         // INPUT
    float  *fixed_image,   // INPUT
    float  *moving_image,  // INPUT
    float  *moving_grad,   // INPUT
    int3   fix_dim,        // Size of fixed image (vox)
    float3 fix_origin,     // Origin of fixed image (mm)
    float3 fix_spacing,    // Spacing of fixed image (mm)
    int3   mov_dim,        // Size of moving image (vox)
    float3 mov_origin,     // Origin of moving image (mm)
    float3 mov_spacing,    // Spacing of moving image (mm)
    int3   roi_dim,        // Dimension of ROI (in vox)
    int3   roi_offset,     // Position of first vox in ROI (in vox)
    int3   vox_per_rgn,    // Knot spacing (in vox)
    int3   rdims,          // # of regions in (x,y,z)
    int3   cdims,
    int    pad,
    float  *skipped        // # of voxels that fell outside the ROI
);

__global__ void sum_reduction_kernel
(
 float *idata, 
 float *odata, 
 int   num_elems);


__global__ void sum_reduction_last_step_kernel
(
 float *idata,
 float *odata,
 int   num_elems);

__global__ void bspline_cuda_update_grad_kernel
(
 float *grad,
 int num_vox,
 int num_elems);


__global__ void kernel_bspline_MI_a_hist_fix (
    float* f_hist_seg,	// partial histogram (moving image)
    float* f_img,	// moving image voxels
    float offset,	// histogram offset
    float delta,	// histogram delta
    long bins,		// # histogram bins
    int3 vpr,		// voxels per region
    int3 fdim,		// fixed  image dimensions
    int3 mdim,		// moving image dimensions
    int3 rdim,		//       region dimensions
    float3 img_origin,	// image origin
    float3 img_spacing,	// image spacing
    float3 mov_offset,	// moving image offset
    float3 mov_ps,	// moving image pixel spacing
    int* c_lut,		// DEBUG
    float* q_lut,	// DEBUG
    float* coeff);	// DEBUG


__global__ void kernel_bspline_MI_a_hist_mov (
    float* m_hist_seg,	// partial histogram (moving image)
    float* m_img,	// moving image voxels
    float offset,	// histogram offset
    float delta,	// histogram delta
    long bins,		// # histogram bins
    int3 vpr,		// voxels per region
    int3 fdim,		// fixed  image dimensions
    int3 mdim,		// moving image dimensions
    int3 rdim,		//       region dimensions
    float3 img_origin,	// image origin
    float3 img_spacing,	// image spacing
    float3 mov_offset,	// moving image offset
    float3 mov_ps,	// moving image pixel spacing
    int* c_lut,	
    float* q_lut,
    float* coeff);


__global__ void kernel_bspline_MI_a_hist_jnt (
    float* skipped,	// OUTPUT:   # of skipped voxels
    float* j_hist,      // OUTPUT:  joint histogram
    float* f_img,	// INPUT:  fixed image voxels
    float* m_img,	// INPUT: moving image voxels
    float f_offset,	// INPUT:  fixed histogram offset 
    float m_offset,	// INPUT: moving histogram offset
    float f_delta,	// INPUT:  fixed histogram delta
    float m_delta,	// INPUT: moving histogram delta
    long f_bins,        // INPUT: #  fixed histogram bins
    long m_bins,	// INPUT: # moving histogram bins
    int3 vpr,		// INPUT: voxels per region
    int3 fdim,		// INPUT:  fixed image dimensions
    int3 mdim,		// INPUT: moving image dimensions
    int3 rdim,		// INPUT: region dimensions
    float3 img_origin,	// INPUT: image origin
    float3 img_spacing,	// INPUT: image spacing
    float3 mov_offset,	// INPUT: moving image offset
    float3 mov_ps,	// INPUT: moving image pixel spacing
    int3 roi_dim,	// INPUT: ROI dimensions
    int3 roi_offset,	// INPUT: ROI Offset
    int* c_lut,		// INPUT: coefficient lut
    float* q_lut,	// INPUT: bspline product lut
    float* coeff);	// INPUT: coefficient array


__global__ void kernel_bspline_MI_a_hist_fix_merge (
 float *f_hist,
 float *f_hist_seg,
 long num_seg_hist);


__global__ void kernel_bspline_MI_hists_a (
    float* skipped,	// OUTPUT:   # of skipped voxels
    float* f_hist,      // OUTPUT:  fixed histogram
    float* m_hist,	// OUTPUT: moving histogram
    float* j_hist,      // OUTPUT:  joint histogram
    float* f_img,	// INPUT:  fixed image voxels
    float* m_img,	// INPUT: moving image voxels
    float f_offset,	// INPUT:  fixed histogram offset 
    float m_offset,	// INPUT: moving histogram offset
    float f_delta,	// INPUT:  fixed histogram delta
    float m_delta,	// INPUT: moving histogram delta
    long f_bins,        // INPUT: #  fixed histogram bins
    long m_bins,	// INPUT: # moving histogram bins
    int3 vpr,		// INPUT: voxels per region
    int3 fdim,		// INPUT:  fixed image dimensions
    int3 mdim,		// INPUT: moving image dimensions
    int3 rdim,		// INPUT: region dimensions
    float3 img_origin,	// INPUT: image origin
    float3 img_spacing,	// INPUT: image spacing
    float3 mov_offset,	// INPUT: moving image offset
    float3 mov_ps,	// INPUT: moving image pixel spacing
    int3 roi_dim,	// INPUT: ROI dimensions
    int3 roi_offset,	// INPUT: ROI Offset
    int* c_lut,		// INPUT: coefficient lut
    float* q_lut,	// INPUT: bspline product lut
    float* coeff);	// INPUT: coefficient array



__global__ void kernel_bspline_MI_dc_dv_a (
    float* dc_dv_x,	// OUTPUT: dC / dv (x-component)
    float* dc_dv_y,	// OUTPUT: dC / dv (y-component)
    float* dc_dv_z,	// OUTPUT: dC / dv (z-component)
    float* f_hist,      // INPUT:  fixed histogram
    float* m_hist,	// INPUT: moving histogram
    float* j_hist,      // INPUT:  joint histogram
    float* f_img,	// INPUT:  fixed image voxels
    float* m_img,	// INPUT: moving image voxels
    float f_offset,	// INPUT:  fixed histogram offset 
    float m_offset,	// INPUT: moving histogram offset
    float f_delta,	// INPUT:  fixed histogram delta
    float m_delta,	// INPUT: moving histogram delta
    long f_bins,        // INPUT: #  fixed histogram bins
    long m_bins,	// INPUT: # moving histogram bins
    int3 vpr,		// INPUT: voxels per region
    int3 fdim,		// INPUT:  fixed image dimensions
    int3 mdim,		// INPUT: moving image dimensions
    int3 rdim,		// INPUT: region dimensions
    float3 img_origin,	// INPUT: image origin
    float3 img_spacing,	// INPUT: image spacing
    float3 mov_offset,	// INPUT: moving image offset
    float3 mov_ps,	// INPUT: moving image pixel spacing
    int3 roi_dim,	// INPUT: ROI dimensions
    int3 roi_offset,	// INPUT: ROI Offset
    int* c_lut,		// INPUT: coefficient lut
    float* q_lut,	// INPUT: bspline product lut
    float* coeff,	// INPUT: coefficient array
    float num_vox_f,	// INPUT: # of voxels
    float score,	// INPUT: evaluated MI cost function
    int pad);		// INPUT: Tile Paddign


#endif
