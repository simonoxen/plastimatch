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
#include "volume.h"
#include "bspline.h"
#include "cuda.h"

/* B-Spline CUDA MI Switches */
//#define MI_HISTS_CPU
//#define MI_GRAD_CPU
#define MI_SCORE_CPU


/* Used by gpu_alloc_copy () */
enum gpu_alloc_copy_mode {
    cudaGlobalMem,
    cudaZeroCopy
};


#if defined __cplusplus
extern "C" {
#endif

// I don't think this is ever used...
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

    // -------------------------------------------------------------------
    // Prototypes: bspline_cuda.cpp 

    void bspline_cuda_MI_a (
		Bspline_parms *parms,
		Bspline_state *bst,
		Bspline_xform *bxf,
		Volume *fixed,
		Volume *moving,
		Volume *moving_grad,
		Dev_Pointers_Bspline *dev_ptrs);





    void bspline_cuda_j_stage_1 (Volume* fixed,
				Volume* moving,
				Volume* moving_grad,
				Bspline_xform* bxf,
				Bspline_parms* parms,
				Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_score_j_mse(Bspline_parms* parms,
			  			Bspline_state *bst,
						Bspline_xform* bxf,
						Volume* fixed,
						Volume* moving,
						Volume* moving_grad,
						Dev_Pointers_Bspline* dev_ptrs);



    void bspline_cuda_score_i_mse(Bspline_parms* parms,
			  			Bspline_state *bst,
						Bspline_xform* bxf,
						Volume* fixed,
						Volume* moving,
						Volume* moving_grad,
						Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_score_h_mse(Bspline_parms* parms,
			  			Bspline_state *bst,
						Bspline_xform* bxf,
						Volume* fixed,
						Volume* moving,
						Volume* moving_grad,
						Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_score_g_mse(
				  Bspline_parms *parms, 
				  Bspline_state *bst,
				  Bspline_xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_f_mse(
				  Bspline_parms *parms, 
				  Bspline_state *bst,
				  Bspline_xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_e_mse_v2(
				     Bspline_parms *parms, 
				     Bspline_state *bst,
				     Bspline_xform* bxf, 
				     Volume *fixed, 
				     Volume *moving, 
				     Volume *moving_grad);

    void bspline_cuda_score_e_mse(
				  Bspline_parms *parms, 
				  Bspline_state *bst,
				  Bspline_xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_d_mse(
				  Bspline_parms *parms, 
				  Bspline_state *bst,
				  Bspline_xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);

    void bspline_cuda_score_c_mse(
				  Bspline_parms *parms, 
				  Bspline_state *bst,
				  Bspline_xform* bxf, 
				  Volume *fixed, 
				  Volume *moving, 
				  Volume *moving_grad);
    //
    // -------------------------------------------------------------------




    // -------------------------------------------------------------------
    // Prototypes: bspline_cuda.cu

    // Simple utility function to check for CUDA runtime errors.
    void checkCUDAError(const char *msg);  

    // Initialize the GPU to execute bspline_cuda_score_j_mse().
    void bspline_cuda_initialize_j (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad,
        Bspline_xform* bxf,
        Bspline_parms* parms
    );

    // Initialize the GPU to execute bspline_cuda_score_i_mse().
    void bspline_cuda_initialize_i(
				   Dev_Pointers_Bspline *dev_ptrs,
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   Bspline_xform *bxf,
				   Bspline_parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_h_mse().
    void bspline_cuda_initialize_h(
				   Dev_Pointers_Bspline *dev_ptrs,
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   Bspline_xform *bxf,
				   Bspline_parms *parms);
    //
    // Initialize the GPU to execute bspline_cuda_score_g_mse().
    void bspline_cuda_initialize_g(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   Bspline_xform *bxf,
				   Bspline_parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_f_mse().
    void bspline_cuda_initialize_f(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   Bspline_xform *bxf,
				   Bspline_parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_e_mse_v2().
    void bspline_cuda_initialize_e_v2(
				      Volume *fixed,
				      Volume *moving,
				      Volume *moving_grad,
				      Bspline_xform *bxf,
				      Bspline_parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_e_mse().
    void bspline_cuda_initialize_e(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   Bspline_xform *bxf,
				   Bspline_parms *parms);

    // Initialize the GPU to execute bspline_cuda_score_d_mse().
    void bspline_cuda_initialize_d(
				   Volume *fixed,
				   Volume *moving,
				   Volume *moving_grad,
				   Bspline_xform *bxf,
				   Bspline_parms *parms);

    // Allocate memory on the GPU and copy all necessary data to the GPU.
    void bspline_cuda_initialize(
				 Volume *fixed,
				 Volume *moving,
				 Volume *moving_grad,
				 Bspline_xform *bxf,
				 Bspline_parms *parms);

    void bspline_cuda_copy_coeff_lut(
				     Bspline_xform *bxf);

    void bspline_cuda_copy_grad_to_host(
					float* host_grad);

    void bspline_cuda_i_stage_1(
		   Volume* fixed,
		   Volume* moving,
		   Volume* moving_grad,
		   Bspline_xform* bxf,
		   Bspline_parms* parms,
		   Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_h_stage_1(
		   Volume* fixed,
		   Volume* moving,
		   Volume* moving_grad,
		   Bspline_xform* bxf,
		   Bspline_parms* parms,
		   Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_calculate_run_kernels_g(
					      Volume *fixed,
					      Volume *moving,
					      Volume *moving_grad,
					      Bspline_xform *bxf,
					      Bspline_parms *parms,
					      int run_low_mem_version, 
					      int debug);

    void bspline_cuda_calculate_run_kernels_f(
					      Volume *fixed,
					      Volume *moving,
					      Volume *moving_grad,
					      Bspline_xform *bxf,
					      Bspline_parms *parms);

    void bspline_cuda_calculate_score_e(
					Volume *fixed,
					Volume *moving,
					Volume *moving_grad,
					Bspline_xform *bxf,
					Bspline_parms *parms);

    void bspline_cuda_run_kernels_e_v2(
				       Volume *fixed,
				       Volume *moving,
				       Volume *moving_grad,
				       Bspline_xform *bxf,
				       Bspline_parms *parms,
				       int sidx0,
				       int sidx1,
				       int sidx2);

    void bspline_cuda_run_kernels_e(
				    Volume *fixed,
				    Volume *moving,
				    Volume *moving_grad,
				    Bspline_xform *bxf,
				    Bspline_parms *parms,
				    int sidx0,
				    int sidx1,
				    int sidx2);

    void bspline_cuda_run_kernels_d(
				    Volume *fixed,
				    Volume *moving,
				    Volume *moving_grad,
				    Bspline_xform *bxf,
				    Bspline_parms *parms,
				    int p0,
				    int p1,
				    int p2);

    void bspline_cuda_run_kernels_c(
				    Volume *fixed,
				    Volume *moving,
				    Volume *moving_grad,
				    Bspline_xform *bxf,
				    Bspline_parms *parms,
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

    void bspline_cuda_j_stage_2(
			Bspline_parms* parms,
			Bspline_xform* bxf,
			Volume* fixed,
			int* vox_per_rgn,
			int* volume_dim,
			float* host_score,
			float* host_grad,
			float* host_grad_mean,
			float* host_grad_norm,
			Dev_Pointers_Bspline* dev_ptrs,
			int* num_vox);


    void bspline_cuda_h_stage_2(
			Bspline_parms* parms,
			Bspline_xform* bxf,
			Volume* fixed,
			int* vox_per_rgn,
			int* volume_dim,
			float* host_score,
			float* host_grad,
			float* host_grad_mean,
			float* host_grad_norm,
			Dev_Pointers_Bspline* dev_ptrs);


    void bspline_cuda_final_steps_f(
				    Bspline_parms* parms, 
				    Bspline_xform* bxf,
				    Volume *fixed,
				    int   *vox_per_rgn,
				    int   *volume_dim,
				    float *host_score,
				    float *host_grad,
				    float *host_grad_mean,
				    float *host_grad_norm);

    void bspline_cuda_final_steps_e_v2(
				       Bspline_parms* parms, 
				       Bspline_xform* bxf,
				       Volume *fixed,
				       int   *vox_per_rgn,
				       int   *volume_dim,
				       float *host_score,
				       float *host_grad,
				       float *host_grad_mean,
				       float *host_grad_norm);

    void bspline_cuda_final_steps_e(
				    Bspline_parms* parms, 
				    Bspline_xform* bxf,
				    Volume *fixed,
				    int   *vox_per_rgn,
				    int   *volume_dim,
				    float *host_score,
				    float *host_grad,
				    float *host_grad_mean,
				    float *host_grad_norm);

    void bspline_cuda_final_steps_d(
				    Bspline_parms* parms, 
				    Bspline_xform* bxf,
				    Volume *fixed,
				    int   *vox_per_rgn,
				    int   *volume_dim,
				    float *host_score,
				    float *host_grad,
				    float *host_grad_mean,
				    float *host_grad_norm);

    void bspline_cuda_calculate_gradient_c (
					    Bspline_parms* parms, 
					    Bspline_state* bst,
					    Bspline_xform* bxf,
					    Volume *fixed,
					    float *host_grad_norm,
					    float *host_grad_mean);

    void bspline_cuda_clean_up_j (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad
    );

    void bspline_cuda_clean_up_i(Dev_Pointers_Bspline* dev_ptrs);

    void bspline_cuda_clean_up_h(Dev_Pointers_Bspline* dev_ptrs);

    void bspline_cuda_clean_up_g();

    void bspline_cuda_clean_up_f();

    void bspline_cuda_clean_up_d();

    void bspline_cuda_clean_up();

    void bspline_cuda_h_push_coeff_lut(Dev_Pointers_Bspline* dev_ptrs, Bspline_xform* bxf);
    
    void bspline_cuda_h_clear_score(Dev_Pointers_Bspline* dev_ptrs);
    
    void bspline_cuda_h_clear_grad(Dev_Pointers_Bspline* dev_ptrs);

    void CUDA_deinterleave( int num_values, float* input, float* out_x, float* out_y, float* out_z);

    void CUDA_pad_64( float** input, int* vol_dims, int* tile_dims);

    void CUDA_pad( float** input, int* vol_dims, int* tile_dims);

    void CUDA_bspline_mse_score_dc_dv( Dev_Pointers_Bspline* dev_ptrs, Bspline_xform* bxf, Volume* fixed, Volume* moving);

    void CUDA_bspline_mse_condense_64_texfetch( Dev_Pointers_Bspline* dev_ptrs, int* vox_per_rgn, int num_tiles);

    void CUDA_bspline_mse_condense_64( Dev_Pointers_Bspline* dev_ptrs, int* vox_per_rgn, int num_tiles);

    void CUDA_bspline_mse_condense( Dev_Pointers_Bspline* dev_ptrs, int* vox_per_rgn, int num_tiles);

    void CUDA_bspline_mse_reduce( Dev_Pointers_Bspline* dev_ptrs, int num_knots);

    float CPU_obtain_spline_basis_function( int t_idx, 
					  int vox_idx, 
					  int vox_per_rgn);
    
    
    void bspline_cuda_init_MI_a ( Dev_Pointers_Bspline* dev_ptrs, Volume* fixed, Volume* moving, Volume* moving_grad, Bspline_xform* bxf, Bspline_parms* parms);

    int CUDA_bspline_MI_a_hist ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);

    void CUDA_bspline_MI_a_hist_fix ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);
    
    void CUDA_bspline_MI_a_hist_mov ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);
    
    int CUDA_bspline_MI_a_hist_jnt ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);

    int CUDA_MI_Hist_a ( BSPLINE_MI_Hist* mi_hist, Bspline_xform *bxf, Volume* fixed, Volume* moving, Dev_Pointers_Bspline *dev_ptrs);
    
    void CUDA_MI_Grad_a ( BSPLINE_MI_Hist* mi_hist, Bspline_state *bst, Bspline_xform *bxf, Volume* fixed, Volume* moving, float num_vox_f, Dev_Pointers_Bspline *dev_ptrs);

    gpuit_EXPORT
    void CUDA_selectgpu (int gpuid);

    void CUDA_listgpu ();

    int CUDA_getarch (int gpuid);

    //
    // -------------------------------------------------------------------

#if defined __cplusplus
}
#endif

#endif
