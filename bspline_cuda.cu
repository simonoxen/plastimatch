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

// Initialize the GPU to execute bspline_cuda_score_d_mse().
void bspline_cuda_initialize_d(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	printf("Initializing CUDA... ");
	fflush(stdout);

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
	fflush(stdout);
}

void bspline_cuda_initialize(
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad,
	BSPLINE_Xform *bxf,
	BSPLINE_Parms *parms)
{
	printf("Initializing CUDA... ");
	fflush(stdout);

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
	fflush(stdout);
}

void bspline_cuda_copy_coeff_lut(
	BSPLINE_Xform *bxf)
{
	// Copy the coefficient LUT to the GPU.
	if(cudaMemcpy(gpu_coeff, bxf->coeff, coeff_mem_size, cudaMemcpyHostToDevice) != cudaSuccess)
		checkCUDAError("Failed to copy coefficient LUT to GPU");
}

void bspline_cuda_clear_score() {
	if(cudaMemset(gpu_score, 0, score_mem_size) != cudaSuccess)
		checkCUDAError("Failed to set score to 0 on GPU\n");
}

void bspline_cuda_clear_grad() {
	if(cudaMemset(gpu_grad, 0, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to set score to 0 on GPU\n");
}

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

	/*
	// Configure the grid.
	int num_elems = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
	int num_blocks = (int)ceil(num_elems / 32.0);
	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(8, 2, 2);
	int smemSize = 32 * sizeof(float);
	// printf("%d thread blocks will be created for each kernel.\n", num_blocks);
	*/
	
	// NAGA: Trying different thread-block sizes
	int num_elems = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
	int num_blocks = (int)ceil(num_elems / 32.0);
	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(32, 1, 1);
	int smemSize = 32 * sizeof(float);
	// printf("%d thread blocks will be created for each kernel.\n", num_blocks);

	// Clear the dc_dv values.
	if(cudaMemset(gpu_dc_dv, 0, dc_dv_mem_size) != cudaSuccess)
		checkCUDAError("cudaMemset failed to fill gpu_dc_dv with 0\n");

	// printf("Launching bspline_cuda_score_d_mse_kernel1... ");
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

	if(cudaThreadSynchronize() != cudaSuccess)
		checkCUDAError("\nbspline_cuda_score_d_mse_kernel1 failed");
	else
		//printf("DONE!\n");

	// Reconfigure the grid.
	num_elems = 192;
	num_blocks = (int)ceil(num_elems / 16.0);
	dim3 dimGrid2(num_blocks, 1, 1);
	dim3 dimBlock2(8, 2, 1);

	// printf("Launching bspline_cuda_score_d_mse_kernel2... ");
	bspline_cuda_score_d_mse_kernel2<<<dimGrid2, dimBlock2>>>(
		gpu_dc_dv,
		gpu_grad,
		gpu_q_lut,
		num_elems,
		p,
		rdims,
		vox_per_rgn
	);

	if(cudaThreadSynchronize() != cudaSuccess)
		checkCUDAError("\nbspline_cuda_score_d_mse_kernel2 failed");
	else
		//printf("DONE!\n");
	
	// Stop the clock.
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	// printf("CUDA kernels for dc_dv and grad completed in %f seconds.\n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);
	fflush(stdout);
}

// Computes the score on the GPU.
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
	fflush(stdout);
}

void bspline_cuda_run_kernels(
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
	fflush(stdout);

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
	fflush(stdout);
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
	fflush(stdout);
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
	fflush(stdout);
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
	fflush(stdout);

	// Copy results back from GPU.
	if(cudaMemcpy(host_dc_dv_x, gpu_dc_dv_x, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
		checkCUDAError("Failed to copy dc_dv stream from GPU to host");
	if(cudaMemcpy(host_dc_dv_y, gpu_dc_dv_y, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
		checkCUDAError("Failed to copy dc_dv stream from GPU to host");
	if(cudaMemcpy(host_dc_dv_z, gpu_dc_dv_z, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
		checkCUDAError("Failed to copy dc_dv stream from GPU to host");

}

void bspline_cuda_calculate_gradient(
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
	fflush(stdout);
}

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
		checkCUDAError("Failed to free memory for dc_dv_x");
	if(cudaFree(gpu_score) != cudaSuccess)
		checkCUDAError("Failed to free memory for dc_dv_y");

	printf("All memory on the GPU has been freed.\n");
}

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

	fflush(stdout);
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err) 
	{
		printf("CUDA Error -- %s: %s.\n", msg, cudaGetErrorString(err));
		fflush(stdout);
		exit(-1);
	} 
}