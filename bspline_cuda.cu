/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_cuda.h"

// Include the kernels.
#include "bspline_cuda_kernels.cu"

// Declare global variables.
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

	// Allocate memory to hold the calculated dc_dv values.
	dc_dv_mem_size = 3 * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2]
		* bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float);
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

	printf("DONE!\n");
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
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

	/*
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
	*/

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
void bspline_cuda_copy_grad_to_host(
	float* host_grad)
{
	if(cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
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
	int run_low_mem_version)
{
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
	
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

	QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

	// Configure the grid.
	int threads_per_block;
	int num_threads;
	int num_blocks;
	int smemSize;

	// printf("Launching bspline_cuda_score_f_mse_kernel1...\n");
	if (!run_low_mem_version) {
		
		printf("Launching one-shot version of bspline_cuda_score_g_mse_kernel1...\n");
		
		threads_per_block = 256;
		num_threads = fixed->npix;
		num_blocks = (int)ceil(num_threads / (float)threads_per_block);
		dim3 dimGrid1(num_blocks / 128, 128, 1);
		dim3 dimBlock1(threads_per_block, 1, 1);
		//smemSize = 4 * sizeof(float) * threads_per_block;
		smemSize = 12 * sizeof(float) * threads_per_block;

		//bspline_cuda_score_g_mse_kernel1_min_regs<<<dimGrid1, dimBlock1>>>(
		bspline_cuda_score_g_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>(
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

	}
	else {

		printf("Launching low memory version of bspline_cuda_score_g_mse_kernel1...\n");

		int tiles_per_launch = 512;

		threads_per_block = 256;
		num_threads = tiles_per_launch * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
		num_blocks = (int)ceil(num_threads / (float)threads_per_block);
		dim3 dimGrid1(num_blocks / 128, 128, 1);
		dim3 dimBlock1(threads_per_block, 1, 1);
		smemSize = 12 * sizeof(float) * threads_per_block;

		for(int i = 0; i < rdims.x * rdims.y * rdims.z; i += tiles_per_launch) {

			bspline_cuda_score_g_mse_kernel1_low_mem<<<dimGrid1, dimBlock1, smemSize>>>(
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

	if(cudaThreadSynchronize() != cudaSuccess)
		checkCUDAError("\bspline_cuda_score_g_mse_compute_score failed");

	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("%f seconds to run bspline_cuda_score_g_mse_kernel1\n", 
		double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

	QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

	// Reconfigure the grid.
	threads_per_block = 256;
	num_threads = bxf->num_knots;
	num_blocks = (int)ceil(num_threads / (float)threads_per_block);
	dim3 dimGrid2(num_blocks, 1, 1);
	dim3 dimBlock2(threads_per_block, 1, 1);
	smemSize = 15 * sizeof(float) * threads_per_block;

	//printf("Launching bspline_cuda_score_f_mse_kernel2...");
	bspline_cuda_score_g_mse_kernel2<<<dimGrid2, dimBlock2, smemSize>>>(
		gpu_dc_dv,
		gpu_grad,
		num_threads,
		rdims,
		cdims,
		vox_per_rgn);

	if(cudaThreadSynchronize() != cudaSuccess)
		checkCUDAError("\bspline_cuda_score_g_mse_kernel2 failed");

	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("%f seconds to run bspline_cuda_score_g_mse_kernel2\n", 
		double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_calculate_run_kernels_f
 *
 * This function runs the kernels to calculate the score and gradient
 * as part of bspline_cuda_score_f_mse.
 ***********************************************************************/
void bspline_cuda_calculate_run_kernels_f(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
	
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

	QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

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

	/*
	// Configure the grid.
	int threads_per_block = 512;
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

	threads_per_block = 512;
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
	*/

	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("%f seconds to run bspline_cuda_score_f_mse_kernel1\n", 
		double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

	QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

	// Reconfigure the grid.
	threads_per_block = 256;
	num_threads = bxf->num_knots;
	num_blocks = (int)ceil(num_threads / (float)threads_per_block);
	dim3 dimGrid2(num_blocks, 1, 1);
	dim3 dimBlock2(threads_per_block, 1, 1);
	int smemSize = 15 * sizeof(float) * threads_per_block;

	//printf("Launching bspline_cuda_score_f_mse_kernel2...");
	bspline_cuda_score_f_mse_kernel2<<<dimGrid2, dimBlock2, smemSize>>>(
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

	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("%f seconds to run bspline_cuda_score_f_mse_kernel2\n", 
		double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_final_steps_f
 *
 * This function performs sum reduction of the score and gradient
 * streams as part of bspline_cuda_score_f_mse.
 ***********************************************************************/
void bspline_cuda_final_steps_f(
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

	if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
		checkCUDAError("Failed to copy grad_norm from GPU to host");
}

/***********************************************************************
 * bspline_cuda_calculate_score_e
 *
 * This function runs the kernel to compute the score values for the
 * entire volume as part of bspline_cuda_score_e_mse.
 ***********************************************************************/
void bspline_cuda_calculate_score_e(
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
void bspline_cuda_run_kernels_e_v2(
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
void bspline_cuda_run_kernels_e(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms,
	int sidx0,
	int sidx1,
	int sidx2)
{
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);

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
void bspline_cuda_final_steps_e_v2(
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
	int3 sdims;
	sdims.x = (int)ceil(bxf->rdims[0] / 4.0);
	sdims.y = (int)ceil(bxf->rdims[1] / 4.0);
	sdims.z = (int)ceil(bxf->rdims[2] / 4.0);

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
void bspline_cuda_final_steps_e(
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
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

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
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	// printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_run_kernels_d
 *
 * This function runs the kernels to compute the score and dc_dv values
 * for a given tile as part of bspline_cuda_score_d_mse.
 ***********************************************************************/
void bspline_cuda_run_kernels_d(
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
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

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
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	// printf("CUDA kernels for dc_dv and grad completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_final_steps_e
 *
 * This function runs the kernels necessary to reduce the score and
 * gradient streams to a single value as part of bspline_cuda_score_d_mse.
 ***********************************************************************/
void bspline_cuda_final_steps_d(
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
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

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
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("CUDA kernels for score completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_run_kernels_c
 *
 * This function runs the kernels necessary to compute the score and
 * dc_dv values as part of bspline_cuda_score_c_mse.
 ***********************************************************************/
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
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

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
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

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
void bspline_cuda_calculate_gradient_c(
	BSPLINE_Parms* parms, 
	BSPLINE_Xform* bxf,
	Volume *fixed,
	float *host_grad_norm,
	float *host_grad_mean) 
{
	BSPLINE_Score* ssd = &parms->ssd;
	
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
	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;
	QueryPerformanceFrequency(&clock_frequency);
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

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
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("CUDA kernels completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
}

/***********************************************************************
 * bspline_cuda_clean_up_g
 *
 * This function frees all allocated memory on the GPU for version "g".
 ***********************************************************************/
void bspline_cuda_clean_up_g() {

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

	printf("All memory on the GPU has been freed.\n");
}

/***********************************************************************
 * bspline_cuda_clean_up_f
 *
 * This function frees all allocated memory on the GPU for version "f".
 ***********************************************************************/
void bspline_cuda_clean_up_f() {

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
void bspline_cuda_clean_up_d() {

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
void bspline_cuda_clean_up() {

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
void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) 
	{
		printf("CUDA Error -- %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	} 
}