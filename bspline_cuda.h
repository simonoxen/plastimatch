/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_cuda_h_
#define _bspline_cuda_h_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "bspline.h"
#include "cuda.h"

#if defined __cplusplus
extern "C" {
#endif

    typedef struct BSPLINE_CUDA_Data_struct BSPLINE_CUDA_Data;
    struct BSPLINE_CUDA_Data_struct {
	size_t image_size;
	float *fixed_image;  // The fixed image
	float *moving_image; // The moving image
	float *moving_grad_x; // Streams to store the gradient of the moving image in the X, Y, and Z directions 
	float *moving_grad_y; 
	float *moving_grad_z; 
	int   *c_lut; // The c_lut indicating which control knots affect voxels within a region
	float *q_lut; // The q_lut indicating the distance of a voxel to each of the 64 control knots
	float *coeff; // The coefficient stream indicating the x, y, z coefficients of each control knot
	float *dx; // Streams to store voxel displacement/gradient values in the X, Y, and Z directions 
	float *dy; 
	float *dz; 
	float *mvr; // Streams to store the mvr values 
	float *diff; // Stream to store the correspondence values in the moving image---for debug purposes only
	float *valid_voxel; // Stream to indicate if a voxel should take part in the score computation or not
	float *partial_sum; // Stream for storing the partial sums during reductions 
	float *sum_element;
    };


    void bspline_cuda_score_h_mse(BSPLINE_Parms* parms,
			  			Bspline_state *bst,
						BSPLINE_Xform* bxf,
						Volume* fixed,
						Volume* moving,
						Volume* moving_grad,
						Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_score_g_mse(
				  BSPLINE_Parms *parms, 
				  Bspline_state *bst,
				  BSPLINE_Xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_f_mse(
				  BSPLINE_Parms *parms, 
				  Bspline_state *bst,
				  BSPLINE_Xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_e_mse_v2(
				     BSPLINE_Parms *parms, 
				     Bspline_state *bst,
				     BSPLINE_Xform* bxf, 
				     Volume *fixed, 
				     Volume *moving, 
				     Volume *moving_grad);

    void bspline_cuda_score_e_mse(
				  BSPLINE_Parms *parms, 
				  Bspline_state *bst,
				  BSPLINE_Xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_d_mse(
				  BSPLINE_Parms *parms, 
				  Bspline_state *bst,
				  BSPLINE_Xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_c_mse(
				  BSPLINE_Parms *parms, 
				  Bspline_state *bst,
				  BSPLINE_Xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    // Simple utility function to check for CUDA runtime errors.
    void checkCUDAError(const char *msg);  

    // Initialize the GPU to execute bspline_cuda_score_g_mse().
    void bspline_cuda_initialize_h(
				   Dev_Pointers_Bspline *dev_ptrs,
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   BSPLINE_Xform *bxf,
				   BSPLINE_Parms *parms);
    //
    // Initialize the GPU to execute bspline_cuda_score_g_mse().
    void bspline_cuda_initialize_g(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   BSPLINE_Xform *bxf,
				   BSPLINE_Parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_f_mse().
    void bspline_cuda_initialize_f(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   BSPLINE_Xform *bxf,
				   BSPLINE_Parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_e_mse_v2().
    void bspline_cuda_initialize_e_v2(
				      Volume *fixed,
				      Volume *moving,
				      Volume *moving_grad,
				      BSPLINE_Xform *bxf,
				      BSPLINE_Parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_e_mse().
    void bspline_cuda_initialize_e(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   BSPLINE_Xform *bxf,
				   BSPLINE_Parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_d_mse().
    void bspline_cuda_initialize_d(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   BSPLINE_Xform *bxf,
				   BSPLINE_Parms *parms);

    // Allocate memory on the GPU and copy all necessary data to the GPU.
    void bspline_cuda_initialize(
				 Volume *fixed,
				 Volume *moving,
				 Volume *moving_grad,
				 BSPLINE_Xform *bxf,
				 BSPLINE_Parms *parms);

    void bspline_cuda_copy_coeff_lut(
				     BSPLINE_Xform *bxf);

    void bspline_cuda_copy_grad_to_host(
					float* host_grad);

    void bspline_cuda_h_stage_1(
		   Volume* fixed,
		   Volume* moving,
		   Volume* moving_grad,
		   BSPLINE_Xform* bxf,
		   BSPLINE_Parms* parms,
		   Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_calculate_run_kernels_g(
					      Volume *fixed,
					      Volume *moving,
					      Volume *moving_grad,
					      BSPLINE_Xform *bxf,
					      BSPLINE_Parms *parms,
					      int run_low_mem_version, 
					      int debug);

    void bspline_cuda_calculate_run_kernels_f(
					      Volume *fixed,
					      Volume *moving,
					      Volume *moving_grad,
					      BSPLINE_Xform *bxf,
					      BSPLINE_Parms *parms);

    void bspline_cuda_calculate_score_e(
					Volume *fixed,
					Volume *moving,
					Volume *moving_grad,
					BSPLINE_Xform *bxf,
					BSPLINE_Parms *parms);

    void bspline_cuda_run_kernels_e_v2(
				       Volume *fixed,
				       Volume *moving,
				       Volume *moving_grad,
				       BSPLINE_Xform *bxf,
				       BSPLINE_Parms *parms,
				       int sidx0,
				       int sidx1,
				       int sidx2);

    void bspline_cuda_run_kernels_e(
				    Volume *fixed,
				    Volume *moving,
				    Volume *moving_grad,
				    BSPLINE_Xform *bxf,
				    BSPLINE_Parms *parms,
				    int sidx0,
				    int sidx1,
				    int sidx2);

    void bspline_cuda_run_kernels_d(
				    Volume *fixed,
				    Volume *moving,
				    Volume *moving_grad,
				    BSPLINE_Xform *bxf,
				    BSPLINE_Parms *parms,
				    int p0,
				    int p1,
				    int p2);

    void bspline_cuda_run_kernels_c(
				    Volume *fixed,
				    Volume *moving,
				    Volume *moving_grad,
				    BSPLINE_Xform *bxf,
				    BSPLINE_Parms *parms,
				    float *host_diff,
				    float *host_dc_dv_x,
				    float *host_dc_dv_y,
				    float *host_dc_dv_z,
				    float *host_score);

    void bspline_cuda_clear_score();

    void bspline_cuda_clear_grad();

    void bspline_cuda_clear_dc_dv();

    void bspline_cuda_compute_score_d(
				      int *vox_per_rgn,
				      int *volume_dim,
				      float *host_score);

    void bspline_cuda_h_stage_2(
			BSPLINE_Parms* parms,
			BSPLINE_Xform* bxf,
			Volume* fixed,
			int* vox_per_rgn,
			int* volume_dim,
			float* host_score,
			float* host_grad,
			float* host_grad_mean,
			float* host_grad_norm,
			Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_final_steps_f(
				    BSPLINE_Parms* parms, 
				    BSPLINE_Xform* bxf,
				    Volume *fixed,
				    int   *vox_per_rgn,
				    int   *volume_dim,
				    float *host_score,
				    float *host_grad,
				    float *host_grad_mean,
				    float *host_grad_norm);

    void bspline_cuda_final_steps_e_v2(
				       BSPLINE_Parms* parms, 
				       BSPLINE_Xform* bxf,
				       Volume *fixed,
				       int   *vox_per_rgn,
				       int   *volume_dim,
				       float *host_score,
				       float *host_grad,
				       float *host_grad_mean,
				       float *host_grad_norm);

    void bspline_cuda_final_steps_e(
				    BSPLINE_Parms* parms, 
				    BSPLINE_Xform* bxf,
				    Volume *fixed,
				    int   *vox_per_rgn,
				    int   *volume_dim,
				    float *host_score,
				    float *host_grad,
				    float *host_grad_mean,
				    float *host_grad_norm);

    void bspline_cuda_final_steps_d(
				    BSPLINE_Parms* parms, 
				    BSPLINE_Xform* bxf,
				    Volume *fixed,
				    int   *vox_per_rgn,
				    int   *volume_dim,
				    float *host_score,
				    float *host_grad,
				    float *host_grad_mean,
				    float *host_grad_norm);

    void bspline_cuda_calculate_gradient_c (
					    BSPLINE_Parms* parms, 
					    Bspline_state* bst,
					    BSPLINE_Xform* bxf,
					    Volume *fixed,
					    float *host_grad_norm,
					    float *host_grad_mean);

    void bspline_cuda_clean_up_h(Dev_Pointers_Bspline* dev_ptrs);

    void bspline_cuda_clean_up_g();

    void bspline_cuda_clean_up_f();

    void bspline_cuda_clean_up_d();

    void bspline_cuda_clean_up();

    void bspline_cuda_h_push_coeff_lut(Dev_Pointers_Bspline* dev_ptrs, BSPLINE_Xform* bxf);
    
    void bspline_cuda_h_clear_score(Dev_Pointers_Bspline* dev_ptrs);
    
    void bspline_cuda_h_clear_grad(Dev_Pointers_Bspline* dev_ptrs);

    void CUDA_deinterleave( int num_values, float* input, float* out_x, float* out_y, float* out_z);

    void CUDA_pad( float** input, int* vol_dims, int* tile_dims);

    void CUDA_bspline_mse_2_condense( Dev_Pointers_Bspline* dev_ptrs, int* vox_per_rgn, int num_tiles);

    void CUDA_bspline_mse_2_reduce( Dev_Pointers_Bspline* dev_ptrs, int num_knots);


    

#if defined __cplusplus
}
#endif

#endif
