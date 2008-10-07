/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_brook_h_
#define _bspline_brook_h_

#include <brook/brook.hpp>

typedef struct BSPLINE_data_on_gpu_struct BSPLINE_DATA_ON_GPU;
struct BSPLINE_data_on_gpu_struct {
    int volume_texture_size; // Size of the volume texture
    int c_lut_texture_size; // Size of the c_lut texture
    int q_lut_texture_size; // Size of the q_lut or multiplier texture
    int coeff_texture_size; // Size of the coefficient texture

    ::brook::stream *fixed_image_stream; // The fixed image
    ::brook::stream *moving_image_stream; // The moving image

	::brook::stream *moving_grad_stream_x; // Streams to store the gradient of the moving image in the X, Y, and Z directions 
	::brook::stream *moving_grad_stream_y; 
	::brook::stream *moving_grad_stream_z; 

    ::brook::stream *c_lut_stream; // The c_lut indicating which control knots affect voxels within a region
    ::brook::stream *q_lut_stream; // The q_lut indicating the distance of a voxel to each of the 64 control knots
    ::brook::stream *coeff_stream; // The coefficient stream indicating the x, y, z coefficients of each control knot

	::brook::stream *dx_stream; // Streams to store voxel displacement/gradient values in the X, Y, and Z directions 
	::brook::stream *dy_stream; 
	::brook::stream *dz_stream; 

	::brook::stream *mvr_stream; // Streams to store the mvr values 
	::brook::stream *diff_stream; // Stream to store the correspondence values in the moving image---for debug purposes only
	::brook::stream *valid_voxel_stream; // Stream to indicate if a voxel should take part in the score computation or not

	::brook::stream *partial_sum_stream; // Stream for storing the partial sums during reductions 
	::brook::stream *sum_element;

	float *dxyz[3];
	float *diff;
	float *valid_voxels;
};

/* Data structure to store the results returned by the GPU */
typedef struct BSPLINE_data_from_gpu_struct BSPLINE_DATA_FROM_GPU;
struct BSPLINE_data_from_gpu_struct {
	float *dxyz[3]; // For the dc_dv values returned from the GPU
	float *diff; // For the diff values returned by the GPU---for debug purposes only
	float *valid_voxels; // Stores a flag for each voxel that indicates if a voxel contributes to the overall score or not 
	float4 score; // The score returned by the GPU
};

#if defined __cplusplus
extern "C" {
#endif
void 
bspline_score_on_gpu_reference(BSPLINE_Parms *parms, 
			       Volume *fixed, Volume *moving, 
			       Volume *moving_grad);


void 
bspline_initialize_streams_on_gpu(Volume* fixed, 
								  Volume* moving, 
								  Volume* moving_grad, 
								  BSPLINE_Parms *parms);

void 
bspline_initialize_structure_to_store_data_from_gpu(Volume* fixed, 
													BSPLINE_Parms *parms);
#if defined __cplusplus
}
#endif

void compute_dxyz_kernel(::brook::stream, 
						 ::brook::stream, 
						 ::brook::stream, 
						 float3, 
						 float3, 
						 float3, 
						 float, 
						 float, 
						 float, 
						 float, 
						 float, 
						 float, 
						 ::brook::stream);

void compute_diff_kernel(::brook::stream, 
						 ::brook::stream, 
						 ::brook::stream, 
						 ::brook::stream, 
						 ::brook::stream, 
						 float3, 
						 float3, 
						 float3, 
						 float3, 
						 float, 
						 float, 
						 float, 
						 ::brook::stream);

void compute_valid_voxels_kernel(::brook::stream, 
								 ::brook::stream, 
								 ::brook::stream, 
								 float3, 
								 float3, 
								 float3, 
								 float3, 
								 float, 
								 ::brook::stream);

void compute_dc_dv_kernel(::brook::stream, 
						  ::brook::stream, 
						  ::brook::stream, 
						  float3, 
						  float3, 
						  float3, 
						  float3, 
						  float, 
						  ::brook::stream);	

void compute_mvr_kernel(::brook::stream, 
						::brook::stream, 
						::brook::stream, 
						 float3, 
						 float3, 
						 float3, 
						 float3, 
						 float, 
						 ::brook::stream);

void compute_diff_squared_kernel(::brook::stream, 
								 ::brook::stream);

/*
void compute_score_kernel(::brook::stream, float4);

void compute_num_valid_voxels_kernel(::brook::stream, 
						  ::brook::stream);
*/


#endif
