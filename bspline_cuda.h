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
#include <cuda.h>
#include "volume.h"
#include "bspline.h"
#include "cuda_mem.h"

/* B-Spline CUDA MI Switches */
//#define MI_HISTS_CPU
//#define MI_GRAD_CPU
#define MI_SCORE_CPU

typedef struct dev_pointers_bspline Dev_Pointers_Bspline;
struct dev_pointers_bspline
{
    // IMPORTANT!
    // Each member of this struct is a POINTER TO
    // AN ADDRESS RESIDING IN THE GPU'S GLOBAL
    // MEMORY!  Care must be taken when referencing
    // and dereferencing members of this structure!

    float* my_gpu_addr;		// Holds address of this
				//   structure in global
				//   device memory.

    float* fixed_image;		// Fixed Image Voxels
    float* moving_image;	// Moving Image Voxels
    float* moving_grad;		// dc_dp (Gradient) Volume

    float* coeff;		// B-Spline coefficients (p)
    float* score;		// The "Score"

    float* f_hist_seg;		// "Segmented" fixed histogram
    float* m_hist_seg;		// "Segmented" moving histogram
    float* j_hist_seg;		// "Segmented" joint histogram

    float* f_hist;		// fixed image histogram
    float* m_hist;		// moving image histogram
    float* j_hist;		// joint histogram

    float* dc_dv;		// dc_dv (Interleaved)
    float* dc_dv_x;		// dc_dv (De-Interleaved)
    float* dc_dv_y;		// dc_dv (De-Interleaved)
    float* dc_dv_z;		// dc_dv (De-Interleaved)

    float* cond_x;		// dc_dv_x (Condensed)
    float* cond_y;		// dc_dv_y (Condensed)
    float* cond_z;		// dc_dv_z (Condensed)

    float* grad;		// dc_dp

    int* LUT_Knot;
    int* LUT_NumTiles;
    int* LUT_Offsets;
    float* LUT_Bspline_x;
    float* LUT_Bspline_y;
    float* LUT_Bspline_z;
    float* skipped;		// # of voxels that fell outside post warp
    unsigned int* skipped_atomic;

    Vmem_Entry* vmem_list;


    // These hold the size of the
    // chucks of memory we allocated
    // that each start at the addresses
    // stored in the pointers above.
    size_t my_size;
    size_t fixed_image_size;
    size_t moving_image_size;
    size_t moving_grad_size;
    size_t coeff_size;
    size_t score_size;
    size_t dc_dv_size;
    size_t dc_dv_x_size;
    size_t dc_dv_y_size;
    size_t dc_dv_z_size;
    size_t cond_x_size;
    size_t cond_y_size;
    size_t cond_z_size;
    size_t grad_size;
    size_t grad_temp_size;
    size_t LUT_Knot_size;
    size_t LUT_NumTiles_size;
    size_t LUT_Offsets_size;
    size_t LUT_Bspline_x_size;
    size_t LUT_Bspline_y_size;
    size_t LUT_Bspline_z_size;
    size_t skipped_size;
    size_t f_hist_size;
    size_t m_hist_size;
    size_t j_hist_size;
    size_t f_hist_seg_size;
    size_t m_hist_seg_size;
    size_t j_hist_seg_size;
};

#if defined __cplusplus
extern "C" {
#endif

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

    void bspline_cuda_clean_up_mse_j (
        Dev_Pointers_Bspline* dev_ptrs,
        Volume* fixed,
        Volume* moving,
        Volume* moving_grad
    );

    void bspline_cuda_clean_up_mi_a (
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

    float CPU_obtain_spline_basis_function( int t_idx, int vox_idx, int vox_per_rgn); 
    
    void bspline_cuda_init_MI_a ( Dev_Pointers_Bspline* dev_ptrs, Volume* fixed, Volume* moving, Volume* moving_grad, Bspline_xform* bxf, Bspline_parms* parms);

    int CUDA_bspline_MI_a_hist ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);

    void CUDA_bspline_MI_a_hist_fix ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);
    
    void CUDA_bspline_MI_a_hist_mov ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);
    
    int CUDA_bspline_MI_a_hist_jnt ( Dev_Pointers_Bspline *dev_ptrs, BSPLINE_MI_Hist* mi_hist, Volume* fixed, Volume* moving, Bspline_xform *bxf);

    int CUDA_MI_Hist_a ( BSPLINE_MI_Hist* mi_hist, Bspline_xform *bxf, Volume* fixed, Volume* moving, Dev_Pointers_Bspline *dev_ptrs);
    
    void CUDA_MI_Grad_a ( BSPLINE_MI_Hist* mi_hist, Bspline_state *bst, Bspline_xform *bxf, Volume* fixed, Volume* moving, float num_vox_f, Dev_Pointers_Bspline *dev_ptrs);

    //
    // -------------------------------------------------------------------

#if defined __cplusplus
}
#endif

#endif
