/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_brook_h_
#define _bspline_brook_h_

#include <brook/brook.hpp>

typedef struct t_s {
	::brook::stream *out_stream;
	::brook::stream *in_stream;
    } TEST_STRUCT;

typedef struct BSPLINE_data_on_gpu_struct BSPLINE_DATA_ON_GPU;
struct BSPLINE_data_on_gpu_struct {
    int volume_texture_size; // Size of the volume texture
    int c_lut_texture_size; // Size of the c_lut texture
    int q_lut_texture_size; // Size of the q_lut or multiplier texture
    int coeff_texture_size; // Size of the coefficient texture
    ::brook::stream *fixed_image_stream; // The fixed image
    ::brook::stream *moving_image_stream; // The moving image
    ::brook::stream *c_lut_stream; // The c_lut indicating which control knots affect voxels within a region
    ::brook::stream *q_lut_stream; // The q_lut indicating the distance of a voxel to each of the 64 control knots
    ::brook::stream *coeff_stream; // The coefficient stream indicating the x, y, z coefficients of each control knot
	::brook::stream *dx_stream; // Stream to store voxel displacement values in the X direction 
	::brook::stream *dy_stream; // Stream to store voxel displacement values in the Y direction
	::brook::stream *dz_stream; // Stream to store voxel displacement values in the Z direction
	::brook::stream *diff_stream; // Stream to store the correspondence values in the moving image---for debug purposes only

	/* Data returned from the GPU */
	float *dxyz[3];
	float *diff;
    };

#if defined __cplusplus
extern "C" {
#endif
void 
bspline_score_on_gpu_reference(BSPLINE_Parms *parms, 
			       Volume *fixed, Volume *moving, 
			       Volume *moving_grad);


void 
bspline_initialize_streams_on_gpu(Volume* fixed, Volume* moving, BSPLINE_Parms *parms);

void toy_a(float loop_index, ::brook::stream result);
void toy_b(::brook::stream in_stream, float loop_index, ::brook::stream result);
void my_sum(::brook::stream foo, ::brook::stream bar);
void init(::brook::stream result);
void compute_dxyz(::brook::stream, ::brook::stream, ::brook::stream, float3, float3, float3, float, float, float, float, float, float, ::brook::stream);
void compute_mvf(::brook::stream, ::brook::stream, ::brook::stream, ::brook::stream, ::brook::stream, float3, float3, float3, float3, float, ::brook::stream);
void compute_diff(::brook::stream, ::brook::stream, ::brook::stream, ::brook::stream, ::brook::stream, 
				   float3, float3, float3, float3, float, float, float, ::brook::stream);

#if defined __cplusplus
}
#endif

#endif
