/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if defined (_WIN32)
#include <windows.h>
#endif
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_cuda.h"
#include "bspline_cuda_kernels.h"

// Define file-scope textures
texture<float, 1, cudaReadModeElementType> tex_fixed_image;
texture<float, 1, cudaReadModeElementType> tex_moving_image;
texture<float, 1, cudaReadModeElementType> tex_moving_grad;
texture<float, 1, cudaReadModeElementType> tex_coeff;
texture<int, 1, cudaReadModeElementType>   tex_c_lut;
texture<float, 1, cudaReadModeElementType> tex_q_lut;
texture<int, 1, cudaReadModeElementType> tex_LUT_Offsets;
texture<float, 1, cudaReadModeElementType> tex_LUT_Bspline_x;
texture<float, 1, cudaReadModeElementType> tex_LUT_Bspline_y;
texture<float, 1, cudaReadModeElementType> tex_LUT_Bspline_z;
texture<float, 1> tex_dc_dv;
texture<float, 1> tex_grad;


////////////////////////////////////////////////////////////
// Note that disabling textures may not
// always work.  Not all GPU kernel functions
// receive a global memory analog of their
// texture references!

#define USE_TEXTURES 1		// Textures Enabled
//#define USE_TEXTURES 0	// Textures Disabled

#if defined (USE_TEXTURES)
#define TEX_REF(array,index) \
    (tex1Dfetch(tex_ ## array, index))
#else
#define TEX_REF(array,index) \
    (array[index])
#endif
////////////////////////////////////////////////////////////


/**
 * A simple kernel used to ensure that CUDA is working correctly.
 *
 * @param dx Stores thread index of every executed thread.
 * @param dy Stores thread index of every executed thread.
 * @param dz Stores thread index of every executed thread.
 */
__global__ void
test_kernel
(
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

extern "C" void bspline_cuda_init_MI_a (
    Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    BSPLINE_Xform* bxf,
    BSPLINE_Parms* parms)
{
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;

    // input volumes
    dev_ptrs->fixed_image_size = fixed->npix * sizeof(float);
    dev_ptrs->moving_image_size = moving->npix * sizeof(float);
    dev_ptrs->moving_grad_size = moving_grad->npix * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->fixed_image, dev_ptrs->fixed_image_size);
    checkCUDAError ("Failed to allocate memory for fixed image");
    cudaMalloc ((void**)&dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    checkCUDAError ("Failed to allocate memory for moving image");
    cudaMalloc ((void**)&dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    checkCUDAError ("Failed to allocate memory for moving grad");
    cudaMemcpy (dev_ptrs->fixed_image, fixed->img, dev_ptrs->fixed_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy (dev_ptrs->moving_image, moving->img, dev_ptrs->moving_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy (dev_ptrs->moving_grad, moving_grad->img, dev_ptrs->moving_grad_size, cudaMemcpyHostToDevice);

    // segmented histograms
    int num_blocks = (fixed->npix + 31) / 32;
    dev_ptrs->f_hist_seg_size = mi_hist->fixed.bins * 2*num_blocks * sizeof(float);
    dev_ptrs->m_hist_seg_size = mi_hist->moving.bins * num_blocks * sizeof(float);
    dev_ptrs->j_hist_seg_size = mi_hist->fixed.bins * num_blocks * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->f_hist_seg, dev_ptrs->f_hist_seg_size);
    checkCUDAError ("Failed to allocate memory for f_hist_seg");
    cudaMalloc ((void**)&dev_ptrs->m_hist_seg, dev_ptrs->m_hist_seg_size);
    checkCUDAError ("Failed to allocate memory for m_hist_seg");
    cudaMalloc ((void**)&dev_ptrs->j_hist_seg, dev_ptrs->j_hist_seg_size);
    checkCUDAError ("Failed to allocate memory for j_hist_seg");


    // histograms
    dev_ptrs->f_hist_size = mi_hist->fixed.bins * sizeof(float);
    dev_ptrs->m_hist_size = mi_hist->moving.bins * sizeof(float);
    dev_ptrs->j_hist_size = mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->f_hist, dev_ptrs->f_hist_size);
    cudaMalloc ((void**)&dev_ptrs->m_hist, dev_ptrs->m_hist_size);
    cudaMalloc ((void**)&dev_ptrs->j_hist, dev_ptrs->j_hist_size);

    // Copy the multiplier LUT to the GPU.
    dev_ptrs->q_lut_size = sizeof(float)
	* bxf->vox_per_rgn[0]
	* bxf->vox_per_rgn[1]
	* bxf->vox_per_rgn[2]
	* 64;

    cudaMalloc((void**)&dev_ptrs->q_lut, dev_ptrs->q_lut_size);
    checkCUDAError("Failed to allocate memory for q_LUT");

    cudaMemcpy(dev_ptrs->q_lut, bxf->q_lut, dev_ptrs->q_lut_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy multiplier q_LUT to GPU");

    cudaBindTexture(0, tex_q_lut, dev_ptrs->q_lut, dev_ptrs->q_lut_size);
    checkCUDAError("Failed to bind tex_q_lut to texture");

    // Copy the index LUT to the GPU.
    dev_ptrs->c_lut_size = sizeof(int) 
	* bxf->rdims[0] 
	* bxf->rdims[1] 
	* bxf->rdims[2] 
	* 64;

    cudaMalloc((void**)&dev_ptrs->c_lut, dev_ptrs->c_lut_size);
    checkCUDAError("Failed to allocate memory for c_LUT");
    cudaMemcpy(dev_ptrs->c_lut, bxf->c_lut, dev_ptrs->c_lut_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy index c_LUT to GPU");
    cudaBindTexture(0, tex_c_lut, dev_ptrs->c_lut, dev_ptrs->c_lut_size);
    checkCUDAError("Failed to bind tex_c_lut to texture");

    dev_ptrs->coeff_size = sizeof(float) * bxf->num_coeff;
    cudaMalloc((void**)&dev_ptrs->coeff, dev_ptrs->coeff_size);
    checkCUDAError("Failed to allocate memory for dev_ptrs->coeff");
    cudaMemset(dev_ptrs->coeff, 0, dev_ptrs->coeff_size);
    cudaBindTexture(0, tex_coeff, dev_ptrs->coeff, dev_ptrs->coeff_size);
    checkCUDAError("Failed to bind dev_ptrs->coeff to texture reference!");

    // score
    dev_ptrs->score_size = sizeof(float) * fixed->npix;
    dev_ptrs->skipped_size = sizeof(float) * fixed->npix;
    cudaMalloc((void**)&dev_ptrs->score, dev_ptrs->score_size);
	
    // grad
    dev_ptrs->grad_size = sizeof(float) * bxf->num_coeff;
    cudaMalloc((void**)&dev_ptrs->grad, dev_ptrs->grad_size);
}


extern "C" void CUDA_bspline_MI_a_hist (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    BSPLINE_Xform* bxf)
{
    // check to see if we get atomic operations
    // for GPU memory
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
    printf ("\n******************* FATAL ERROR *******************\n");
    printf ("   Atomic memory operations not supported by GPU!\n");
    printf ("     A GPU of Compute Capability 1.2 or greater\n");
    printf ("     is required to for GPU accelerated MI\n");
    printf ("***************************************************\n\n");
    exit(0);
#endif

    // Generate the fixed histogram (48 ms)
    CUDA_bspline_MI_a_hist_fix (dev_ptrs, mi_hist, fixed);
    cudaMemcpy (mi_hist->f_hist, dev_ptrs->f_hist, dev_ptrs->f_hist_size, cudaMemcpyDeviceToHost);

    // Generate the moving histogram (150 ms)
    CUDA_bspline_MI_a_hist_mov (dev_ptrs, mi_hist, fixed, moving, bxf);
    cudaMemcpy (mi_hist->m_hist, dev_ptrs->m_hist, dev_ptrs->m_hist_size, cudaMemcpyDeviceToHost);

    // Generate the joint histogram (??.?? ms) -- not written
    CUDA_bspline_MI_a_hist_jnt (dev_ptrs, mi_hist, fixed, moving, bxf);
    cudaMemcpy (mi_hist->j_hist, dev_ptrs->j_hist, dev_ptrs->j_hist_size, cudaMemcpyDeviceToHost);
}



extern "C" void CUDA_bspline_MI_a_hist_fix (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed)
{
    // Initialize histogram memory on GPU
    cudaMemset(dev_ptrs->f_hist_seg, 0, dev_ptrs->f_hist_seg_size);
    cudaMemset(dev_ptrs->f_hist, 0, dev_ptrs->f_hist_size);

    // --- INITIALIZE GRID ---
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 32;
    int num_threads = fixed->npix;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    int smemSize = threads_per_block * mi_hist->fixed.bins * sizeof(float);

    // -----
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // -----


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	printf("\n[ERROR] Unable to find suitable kernel_bspline_MI_a_hist_fix() configuration!\n");
	exit(0);
    } else {
#if defined (commentout)
	printf ("Grid [%i,%i], %d threads_per_block.\n", 
	    Grid_x, Grid_y, threads_per_block);
#endif
    }

    dim3 dimGrid1(Grid_x, Grid_y, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);
    // ----------------------

    // Launch kernel with one thread per voxel
    kernel_bspline_MI_a_hist_fix <<<dimGrid1, dimBlock1, smemSize>>> (
	dev_ptrs->f_hist_seg,
	dev_ptrs->fixed_image,
	mi_hist->fixed.offset,
	1.0f/mi_hist->fixed.delta,
	mi_hist->fixed.bins,
	threads_per_block);
					
    checkCUDAError ("kernel hist_fix");
    int num_sub_hists = num_blocks;


    // Merge sub-histograms
    threads_per_block = 512;
    dim3 dimGrid2 (mi_hist->fixed.bins, 1, 1);
    dim3 dimBlock2 (threads_per_block, 1, 1);
    smemSize = 512 * sizeof(float);
	
    // this kernel can be ran with any thread-block size that
    // contains a power of 2 # threads.
    kernel_bspline_MI_a_hist_fix_merge <<<dimGrid2 , dimBlock2, smemSize>>> (
	dev_ptrs->f_hist,
	dev_ptrs->f_hist_seg,
	num_sub_hists);

    checkCUDAError ("kernel hist_fix_merge");
}


extern "C" void CUDA_bspline_MI_a_hist_mov (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    BSPLINE_Xform *bxf)
{
    // Initialize histogram memory on GPU
    cudaMemset(dev_ptrs->m_hist_seg, 0, dev_ptrs->m_hist_seg_size);
    cudaMemset(dev_ptrs->m_hist, 0, dev_ptrs->m_hist_size);

    int3 vpr;
    vpr.x = bxf->vox_per_rgn[0];
    vpr.y = bxf->vox_per_rgn[1];
    vpr.z = bxf->vox_per_rgn[2];

    int3 fdim;
    fdim.x = fixed->dim[0];
    fdim.y = fixed->dim[1];
    fdim.z = fixed->dim[2];

    int3 mdim;
    mdim.x = moving->dim[0]; mdim.y = moving->dim[1];
    mdim.z = moving->dim[2];
	
    int3 rdim;
    rdim.x = bxf->rdims[0];
    rdim.y = bxf->rdims[1];
    rdim.z = bxf->rdims[2];

    float3 img_origin;
    img_origin.x = bxf->img_origin[0];
    img_origin.y = bxf->img_origin[1];
    img_origin.z = bxf->img_origin[2];
	
    float3 img_spacing;     
    img_spacing.x = bxf->img_spacing[0];
    img_spacing.y = bxf->img_spacing[1];
    img_spacing.z = bxf->img_spacing[2];


    float3 mov_offset;     
    mov_offset.x = moving->offset[0];
    mov_offset.y = moving->offset[1];
    mov_offset.z = moving->offset[2];

    float3 mov_ps;
    mov_ps.x = moving->pix_spacing[0];
    mov_ps.y = moving->pix_spacing[1];
    mov_ps.z = moving->pix_spacing[2];
	

    // --- INITIALIZE GRID ---
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 32;
    int num_threads = fixed->npix;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    int smemSize = threads_per_block * mi_hist->fixed.bins * sizeof(float);

    // -----
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // -----


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	printf("\n[ERROR] Unable to find suitable kernel_bspline_MI_a_hist_mov() configuration!\n");
	exit(0);
    } else {
#if defined (commentout)
	printf ("Grid [%i,%i], %d threads_per_block.\n", 
	    Grid_x, Grid_y, threads_per_block);
#endif
    }

    dim3 dimGrid1(Grid_x, Grid_y, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);
    //	printf ("  -- GRID: %i, %i\n", Grid_x, Grid_y);
    // ----------------------

    // Launch kernel with one thread per voxel
    kernel_bspline_MI_a_hist_mov <<<dimGrid1, dimBlock1, smemSize>>> (
	dev_ptrs->m_hist_seg,		// partial histogram (moving image)
	dev_ptrs->fixed_image,		// fixed  image voxels
	dev_ptrs->moving_image,		// moving image voxels
	mi_hist->moving.offset,		// histogram offset
	1.0f/mi_hist->moving.delta,	// histogram delta
	mi_hist->moving.bins,		// # histogram bins
	vpr,				// voxels per region
	fdim,				// fixed  image dimensions
	mdim,				// moving image dimensions
	rdim,				//       region dimensions
	img_origin,			// image origin
	img_spacing,			// image spacing
	mov_offset,			// moving image offset
	mov_ps,				// moving image pixel spacing
	dev_ptrs->c_lut,		// DEBUG
	dev_ptrs->q_lut,		// DEBUG
	dev_ptrs->coeff,		// DEBUG
	threads_per_block);		// # threads (to be removed)

    checkCUDAError ("kernel hist_mov");

    int num_sub_hists = num_blocks;


    // Merge sub-histograms
    threads_per_block = 512;
    dim3 dimGrid2 (mi_hist->fixed.bins, 1, 1);
    dim3 dimBlock2 (threads_per_block, 1, 1);
    smemSize = 512 * sizeof(float);
	
    // this kernel can be ran with any thread-block size
    kernel_bspline_MI_a_hist_fix_merge <<<dimGrid2 , dimBlock2, smemSize>>> (
	dev_ptrs->m_hist,
	dev_ptrs->m_hist_seg,
	num_sub_hists);

    checkCUDAError ("kernel hist_mov_merge");
}


extern "C" void CUDA_bspline_MI_a_hist_jnt (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    BSPLINE_Xform *bxf)
{
    // Initialize histogram memory on GPU
    cudaMemset(dev_ptrs->j_hist_seg, 0, dev_ptrs->j_hist_seg_size);
    cudaMemset(dev_ptrs->j_hist, 0, dev_ptrs->j_hist_size);

    int3 vpr;
    vpr.x = bxf->vox_per_rgn[0];
    vpr.y = bxf->vox_per_rgn[1];
    vpr.z = bxf->vox_per_rgn[2];

    int3 fdim;
    fdim.x = fixed->dim[0];
    fdim.y = fixed->dim[1];
    fdim.z = fixed->dim[2];

    int3 mdim;
    mdim.x = moving->dim[0];
    mdim.y = moving->dim[1];
    mdim.z = moving->dim[2]; 
    int3 rdim;
    rdim.x = bxf->rdims[0];
    rdim.y = bxf->rdims[1];
    rdim.z = bxf->rdims[2];

    float3 img_origin;
    img_origin.x = bxf->img_origin[0];
    img_origin.y = bxf->img_origin[1];
    img_origin.z = bxf->img_origin[2];
	
    float3 img_spacing;     
    img_spacing.x = bxf->img_spacing[0];
    img_spacing.y = bxf->img_spacing[1];
    img_spacing.z = bxf->img_spacing[2];


    float3 mov_offset;     
    mov_offset.x = moving->offset[0];
    mov_offset.y = moving->offset[1];
    mov_offset.z = moving->offset[2];

    float3 mov_ps;
    mov_ps.x = moving->pix_spacing[0];
    mov_ps.y = moving->pix_spacing[1];
    mov_ps.z = moving->pix_spacing[2];
	
    int num_bins = (int)mi_hist->fixed.bins * (int)mi_hist->moving.bins;

    // --- INITIALIZE GRID ---
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 32;
    int num_threads = fixed->npix;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    int smemSize = num_bins * sizeof(float);

    // -----
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // -----


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	printf("\n[ERROR] Unable to find suitable kernel_bspline_MI_a_hist_jnt() configuration!\n");
	exit(0);
    } else {
#if defined (commentout)
	printf ("Grid [%i,%i], %d threads_per_block.\n", 
	    Grid_x, Grid_y, threads_per_block);
#endif
    }

    dim3 dimGrid1(Grid_x, Grid_y, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);
    //	printf ("  -- GRID: %i, %i\n", Grid_x, Grid_y);
    // ----------------------

    // Launch kernel with one thread per voxel
    kernel_bspline_MI_a_hist_jnt <<<dimGrid1, dimBlock1, smemSize>>> (
	dev_ptrs->j_hist,		// partial histogram (moving image)
	//			dev_ptrs->j_hist_seg,		// partial histogram (moving image)
	dev_ptrs->fixed_image,		// fixed  image voxels
	dev_ptrs->moving_image,		// moving image voxels
	mi_hist->fixed.offset,		// fixed histogram offset
	mi_hist->moving.offset,		// moving histogram offset
	1.0f/mi_hist->fixed.delta,	// fixed histogram delta
	1.0f/mi_hist->moving.delta,	// moving histogram delta
	mi_hist->moving.bins,		// # moving bins
	num_bins,	 		// # joint bins
	vpr,				// voxels per region
	fdim,				// fixed  image dimensions
	mdim,				// moving image dimensions
	rdim,				//       region dimensions
	img_origin,			// image origin
	img_spacing,			// image spacing
	mov_offset,			// moving image offset
	mov_ps,				// moving image pixel spacing
	dev_ptrs->c_lut,		// DEBUG
	dev_ptrs->q_lut,		// DEBUG
	dev_ptrs->coeff,		// DEBUG
	threads_per_block);		// # threads (to be removed)

    checkCUDAError ("kernel hist_mov");

    /*
      int num_sub_hists = num_blocks;

      // Merge sub-histograms
      threads_per_block = 512;
      dim3 dimGrid2 (num_bins, 1, 1);
      dim3 dimBlock2 (threads_per_block, 1, 1);
      smemSize = 512 * sizeof(float);
	
      // this kernel can be ran with any thread-block size
      kernel_bspline_MI_a_hist_fix_merge <<<dimGrid2 , dimBlock2, smemSize>>> (
      dev_ptrs->j_hist,
      dev_ptrs->j_hist_seg,
      num_sub_hists);

      checkCUDAError ("kernel hist_jnt_merge");
    */
					
}



////////////////////////////////////////////////////////////////////////////////
// Generates many sub-histograms of the fixed image
//
// NOTE: The main focus of this kernel is to avoid shared memory
//       bank conflicts.
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bspline_MI_a_hist_fix (
    float* f_hist_seg,
    float* fixed,
    float offset,
    float delta,
    long bins,
    int nthreads)
{
    int bin;
    int stride;


    // -- Setup Thread Attributes -----------------------------
    int blockIdxInGrid = (gridDim.x * blockIdx.y) + blockIdx.x;
    int blockOffset = blockIdxInGrid * blockDim.x;
    int voxel_idx = blockOffset + threadIdx.x;
    // --------------------------------------------------------


    // -- Initialize Shared Memory ----------------------------
    extern __shared__ float s_Fixed[];

    for (long i=0; i < bins; i++)
	s_Fixed[threadIdx.x + i*nthreads] = 0.0f;
    // --------------------------------------------------------

    __syncthreads();
	
    // -- Accumulate Into Segmented Histograms ----------------
    for (int chunk=0; chunk < 1; chunk++)
    {
	stride = chunk * 64;
	bin = (int)((fixed[voxel_idx + stride] - offset) * delta);
	s_Fixed[threadIdx.x + bin*nthreads]++;
    }
    // --------------------------------------------------------

    __syncthreads();

    // -- Merge Segmented Histograms --------------------------
    if (threadIdx.x < bins)
    {
	float sum = 0.0f;


	// Stagger the starting shared memory bank
	// access for each thread so as to prevent
	// bank conflicts, which reasult in half
	// warp difergence / serialization.
	const int startPos = (threadIdx.x & 0x0F);
	const int offset   = threadIdx.x * nthreads;

	for (int i=0, accumPos = startPos; i < nthreads; i++)
	{
	    sum += s_Fixed[offset + accumPos];
	    if (++accumPos == nthreads)
		accumPos = 0;
	}

	f_hist_seg[blockIdxInGrid*bins + threadIdx.x] = sum;

    }
    // --------------------------------------------------------

    // Done.
    // We now have (num_thread_blocks) partial histograms that
    // need to be merged.  This will be done with another
    // kernel to be ran immediately following the completion
    // of this kernel.
}




////////////////////////////////////////////////////////////////////////////////
// Generates many sub-histograms of the moving image
//
//                 --- Neightborhood of 6 ---
//
// NOTE: The main focus of this kernel is to avoid shared memory
//       bank conflicts.
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bspline_MI_a_hist_mov (
    float* m_hist_seg,	// partial histogram (moving image)
    float* fixed,		// fixed  image voxels
    float* moving,		// moving image voxels
    float offset,		// histogram offset
    float delta,		// histogram delta
    long bins,		// # histogram bins
    int3 vpr,		// voxels per region
    int3 fdim,		// fixed  image dimensions
    int3 mdim,		// moving image dimensions
    int3 rdim,		//       region dimensions
    float3 img_origin,	// image origin
    float3 img_spacing,	// image spacing
    float3 mov_offset,	// moving image offset
    float3 mov_ps,		// moving image pixel spacing
    int* c_lut,		// DEBUG
    float* q_lut,		// DEBUG
    float* coeff,		// DEBUG
    int nthreads)		// # threads (to be removed)
{
    // -- Initialize Shared Memory ----------------------------
    // Amount: 32 * # bins
    extern __shared__ float s_Moving[];

    for (long i=0; i < bins; i++)
	s_Moving[threadIdx.x + i*nthreads] = 0.0f;
    // --------------------------------------------------------


    __syncthreads();


    // -- Setup Thread Attributes -----------------------------
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;
    // --------------------------------------------------------

	
    // -- Only process threads that map to voxels -------------
    if (thread_idxg > fdim.x * fdim.y * fdim.z)
	return;
    // --------------------------------------------------------


    // -- Variables used by histogram -------------------------
    long bin;
    //	int  stride;
    // --------------------------------------------------------


    // -- Variables used by correspondence --------------------
    // -- (Block verified) ------------------------------------
    int3 r;			// Voxel index (global)
    int4 q;			// Voxel index (local)
    int4 p;			// Tile index


    float3 f;		// Distance from origin (in mm )
    float3 m;		// Voxel Displacement   (in mm )
    float3 n;		// Voxel Displacement   (in vox)
    float3 d;		// Deformation vector

    int3 miqs;		// PVI - 6 NBH
    int3 mjqs;		//    Moving image pixel coords
    int3 mkqs;

    float3 fxqs;		// PVI - 6 NBH
    float3 fyqs;		//    Interpolant fraction
    float3 fzqs;
    int mvf;


    //   ----    ----    ----    ----    ----    ----    ----    
	
    r.z = thread_idxg / (fdim.x * fdim.y);
    r.y = (thread_idxg - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = thread_idxg - r.z * fdim.x * fdim.y - (r.y * fdim.x);
	
    p.x = r.x / vpr.x;
    p.y = r.y / vpr.y;
    p.z = r.z / vpr.z;
    p.w = ((p.z * rdim.y + p.y) * rdim.x) + p.x;

    q.x = r.x - p.x * vpr.x;
    q.y = r.y - p.y * vpr.y;
    q.z = r.z - p.z * vpr.z;
    q.w = ((q.z * vpr.y * q.y) * vpr.x) + q.x;

    f.x = img_origin.x + img_spacing.x * r.x;
    f.y = img_origin.y + img_spacing.y * r.y;
    f.z = img_origin.z + img_spacing.z * r.z;
    // --------------------------------------------------------


    // -- Compute deformation vector --------------------------
    int cidx;
    float P;

    d.x = 0.0f;
    d.y = 0.0f;
    d.z = 0.0f;

    for (int k=0; k < 64; k++)
    {
	// Texture Version
	P = tex1Dfetch (tex_q_lut, 64*q.w + k);
	cidx = 3 * tex1Dfetch (tex_c_lut, 64*p.w + k);

	d.x += P * tex1Dfetch (tex_coeff, cidx + 0);
	d.y += P * tex1Dfetch (tex_coeff, cidx + 1);
	d.z += P * tex1Dfetch (tex_coeff, cidx + 2);


	// Global Memory Version
	//		P = q_lut[64*q.w + k];
	//		cidx = 3 * c_lut[64*p.w + k];
	//
	//		d.x += P * coeff[cidx + 0];
	//		d.y += P * coeff[cidx + 1];
	//		d.z += P * coeff[cidx + 2];
    }
    // --------------------------------------------------------


    // -- Correspondence --------------------------------------
    // -- (Block verified) ------------------------------------
    m.x = f.x + d.x;
    m.y = f.y + d.y;
    m.z = f.z + d.z;

    // n.x = m.i  etc
    n.x = (m.x - mov_offset.x) / mov_ps.x;
    n.y = (m.y - mov_offset.y) / mov_ps.y;
    n.z = (m.z - mov_offset.z) / mov_ps.z;

    if (n.x < -0.5 || n.x > mdim.x - 0.5 ||
	n.y < -0.5 || n.y > mdim.y - 0.5 ||
	n.z < -0.5 || n.z > mdim.z - 0.5)
    {
	// -->> skipped voxel logic here <<--
	return;
    }
    // --------------------------------------------------------


    // -- Compute quadratic interpolation fractions -----------
    float t, t2, t22;
    float marf;
    long mari;

    // --- - x - ---
    marf = (float)(n.x + 0.5);
    mari = (long)(n.x + 0.5);
    t = n.x - marf;
    t2 = t * t;
    t22 = 0.5 * t2;

    // Generate fxqs
    fxqs.x = t22;
    //	fxqs.y = 1 - (t2 + t + 0.5);
    fxqs.y = - t2 + t + 0.5;
    fxqs.z = t22 - t + 0.5;

    // Generate miqs
    miqs.x = mari - 1;
    miqs.y = mari;
    miqs.z = mari + 1;

    // --- - y - ---
    marf = (float)(n.y + 0.5);
    mari = (long)(n.y + 0.5);
    t = n.y - marf;
    t2 = t * t;
    t22 = 0.5 * t2;

    // Generate fxqs
    fyqs.x = t22;
    //	fyqs.y = 1 - (t2 + t + 0.5);
    fyqs.y = - t2 + t + 0.5;
    fyqs.z = t22 - t + 0.5;

    // Generate miqs
    mjqs.x = mari - 1;
    mjqs.y = mari;
    mjqs.z = mari + 1;

	
    // --- - z - ---
    marf = (float)(n.z + 0.5);
    mari = (long)(n.z + 0.5);
    t = n.z - marf;
    t2 = t * t;
    t22 = 0.5 * t2;

    // Generate fxqs
    fzqs.x = t22;
    //	fzqs.y = 1 - (t2 + t + 0.5);
    fzqs.y = - t2 + t + 0.5;
    fzqs.z = t22 - t + 0.5;

    // Generate miqs
    mkqs.x = mari - 1;
    mkqs.y = mari;
    mkqs.z = mari + 1;

    // --- Bounds checking
    if (miqs.x < 0) miqs.x = 0;
    if (miqs.y < 0) miqs.y = 0;
    if (miqs.z < 0) miqs.z = 0;
    if (mjqs.x < 0) mjqs.x = 0;	
    if (mjqs.y < 0) mjqs.y = 0;
    if (mjqs.z < 0) mjqs.z = 0;
    if (mkqs.x < 0) mkqs.x = 0;
    if (mkqs.y < 0) mkqs.y = 0;
    if (mkqs.z < 0) mkqs.z = 0;
    // --------------------------------------------------------

    __syncthreads();

    // -- Accumulate Into Segmented Histograms ----------------
    float midx;
    float mf_1, mf_2;
    float amt;



    // --- -- - BIN #1 - -- ---
    mvf = (mkqs.y * mdim.y + mjqs.y) * mdim.x + miqs.y;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fxqs.y + fyqs.y + fzqs.y);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;

    // --- -- - BIN #2 - -- ---
    mvf = (mkqs.y * mdim.y + mjqs.y) * mdim.x + miqs.x;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((int)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fxqs.x);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;

    // --- -- - BIN #3 - -- ---
    mvf = (mkqs.y * mdim.y + mjqs.y) * mdim.x + miqs.z;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((int)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fxqs.z);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;

    // --- -- - BIN #4 - -- ---
    mvf = (mkqs.y * mdim.y + mjqs.x) * mdim.x + miqs.y;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((int)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fyqs.x);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;

    // --- -- - BIN #5 - -- ---
    mvf = (mkqs.y * mdim.y + mjqs.z) * mdim.x + miqs.y;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((int)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fyqs.z);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;

    // --- -- - BIN #6 - -- ---
    mvf = (mkqs.x * mdim.y + mjqs.y) * mdim.x + miqs.y;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((int)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fzqs.x);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;

    // --- -- - BIN #7 - -- ---
    mvf = (mkqs.z * mdim.y + mjqs.y) * mdim.x + miqs.y;
    midx = (moving[mvf] - offset) * delta;
    bin = (long)(midx);
    mf_1 = midx - (float)((int)midx);
    mf_2 = 1.0f - mf_1;
    amt = (1.0/3.0) * (fzqs.z);
    s_Moving[threadIdx.x + bin*nthreads] += mf_1 * amt;
    s_Moving[threadIdx.x + (bin+1)*nthreads] += mf_2 * amt;
    // --------------------------------------------------------

    __syncthreads();

    // -- Merge Segmented Histograms --------------------------
    if (threadIdx.x < bins)
    {
	float sum = 0.0f;


	// Stagger the starting shared memory bank
	// access for each thread so as to prevent
	// bank conflicts, which reasult in half
	// warp difergence / serialization.
	const int startPos = (threadIdx.x & 0x0F);
	const int offset   = threadIdx.x * nthreads;

	for (int i=0, accumPos = startPos; i < nthreads; i++)
	{
	    sum += s_Moving[offset + accumPos];
	    if (++accumPos == nthreads)
		accumPos = 0;
	}

	m_hist_seg[blockIdxInGrid*bins + threadIdx.x] = sum;

    }
    // --------------------------------------------------------

    // Done.
    // We now have (num_thread_blocks) partial histograms that
    // need to be merged.  This will be done with another
    // kernel to be ran immediately following the completion
    // of this kernel.

    //NOTE:
    // fv = thread_idxg
    // fi = r.x
    // fj = r.y
    // fk = r.z
}




////////////////////////////////////////////////////////////////////////////////
// Generates the joint histogram
//
//                 --- Neightborhood of 6 ---
//
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bspline_MI_a_hist_jnt (
    float* j_hist_seg,	// partial histogram (joint)
    float* fixed,		// fixed  image voxels
    float* moving,		// moving image voxels
    float f_offset,		// fixed histogram offset
    float m_offset,		// moving histogram offset
    float f_delta,		// fixed histogram delta
    float m_delta,		// moving histogram delta
    long m_bins,		// # moving histogram bins
    long j_bins,		// # joint  histogram bins
    int3 vpr,		// voxels per region
    int3 fdim,		// fixed  image dimensions
    int3 mdim,		// moving image dimensions
    int3 rdim,		//       region dimensions
    float3 img_origin,	// image origin
    float3 img_spacing,	// image spacing
    float3 mov_offset,	// moving image offset
    float3 mov_ps,		// moving image pixel spacing
    int* c_lut,		// DEBUG
    float* q_lut,		// DEBUG
    float* coeff,		// DEBUG
    int nthreads)		// # threads (to be removed)
{
    // -- Initialize Shared Memory ----------------------------
    // Amount: (# fixed bins) * (# moving bins)
    extern __shared__ float s_Joint[];

    long clusters = (j_bins + 31) / 32;

    for (long i=0; i < clusters; i++)
	if (threadIdx.x + 32*i < j_bins)
	    s_Joint[threadIdx.x + 32*i] = 0.0f;
    // --------------------------------------------------------


    __syncthreads();


    // -- Setup Thread Attributes -----------------------------
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;
    // --------------------------------------------------------

	
    // -- Only process threads that map to voxels -------------
    if (thread_idxg > fdim.x * fdim.y * fdim.z)
	return;
    // --------------------------------------------------------


    // -- Variables used by correspondence --------------------
    // -- (Block verified) ------------------------------------
    int3 r;			// Voxel index (global)
    int4 q;			// Voxel index (local)
    int4 p;			// Tile index


    float3 f;		// Distance from origin (in mm )
    float3 m;		// Voxel Displacement   (in mm )
    float3 n;		// Voxel Displacement   (in vox)
    float3 d;		// Deformation vector

    int3 miqs;		// PVI - 6 NBH
    int3 mjqs;		//    Moving image pixel coords
    int3 mkqs;

    float3 fxqs;		// PVI - 6 NBH
    float3 fyqs;		//    Interpolant fraction
    float3 fzqs;
    int mvf;


    //   ----    ----    ----    ----    ----    ----    ----    
	
    r.z = thread_idxg / (fdim.x * fdim.y);
    r.y = (thread_idxg - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = thread_idxg - r.z * fdim.x * fdim.y - (r.y * fdim.x);
	
    p.x = r.x / vpr.x;
    p.y = r.y / vpr.y;
    p.z = r.z / vpr.z;
    p.w = ((p.z * rdim.y + p.y) * rdim.x) + p.x;

    q.x = r.x - p.x * vpr.x;
    q.y = r.y - p.y * vpr.y;
    q.z = r.z - p.z * vpr.z;
    q.w = ((q.z * vpr.y * q.y) * vpr.x) + q.x;

    f.x = img_origin.x + img_spacing.x * r.x;
    f.y = img_origin.y + img_spacing.y * r.y;
    f.z = img_origin.z + img_spacing.z * r.z;
    // --------------------------------------------------------


    // -- Compute deformation vector --------------------------
    int cidx;
    float P;

    d.x = 0.0f;
    d.y = 0.0f;
    d.z = 0.0f;

    for (int k=0; k < 64; k++)
    {
	// Texture Version
	P = tex1Dfetch (tex_q_lut, 64*q.w + k);
	cidx = 3 * tex1Dfetch (tex_c_lut, 64*p.w + k);

	d.x += P * tex1Dfetch (tex_coeff, cidx + 0);
	d.y += P * tex1Dfetch (tex_coeff, cidx + 1);
	d.z += P * tex1Dfetch (tex_coeff, cidx + 2);


	// Global Memory Version
	//		P = q_lut[64*q.w + k];
	//		cidx = 3 * c_lut[64*p.w + k];
	//
	//		d.x += P * coeff[cidx + 0];
	//		d.y += P * coeff[cidx + 1];
	//		d.z += P * coeff[cidx + 2];
    }
    // --------------------------------------------------------


    // -- Correspondence --------------------------------------
    // -- (Block verified) ------------------------------------
    m.x = f.x + d.x;
    m.y = f.y + d.y;
    m.z = f.z + d.z;

    // n.x = m.i  etc
    n.x = (m.x - mov_offset.x) / mov_ps.x;
    n.y = (m.y - mov_offset.y) / mov_ps.y;
    n.z = (m.z - mov_offset.z) / mov_ps.z;

    if (n.x < -0.5 || n.x > mdim.x - 0.5 ||
	n.y < -0.5 || n.y > mdim.y - 0.5 ||
	n.z < -0.5 || n.z > mdim.z - 0.5)
    {
	// -->> skipped voxel logic here <<--
	return;
    }
    // --------------------------------------------------------


    // -- Compute quadratic interpolation fractions -----------
    float t, t2, t22;
    float marf;
    long mari;

    // --- - x - ---
    marf = (float)(n.x + 0.5);
    mari = (long)(n.x + 0.5);
    t = n.x - marf;
    t2 = t * t;
    t22 = 0.5 * t2;

    // Generate fxqs
    fxqs.x = t22;
    //	fxqs.y = 1 - (t2 + t + 0.5);
    fxqs.y = - t2 + t + 0.5;
    fxqs.z = t22 - t + 0.5;

    // Generate miqs
    miqs.x = mari - 1;
    miqs.y = mari;
    miqs.z = mari + 1;

    // --- - y - ---
    marf = (float)(n.y + 0.5);
    mari = (long)(n.y + 0.5);
    t = n.y - marf;
    t2 = t * t;
    t22 = 0.5 * t2;

    // Generate fxqs
    fyqs.x = t22;
    //	fyqs.y = 1 - (t2 + t + 0.5);
    fyqs.y = - t2 + t + 0.5;
    fyqs.z = t22 - t + 0.5;

    // Generate miqs
    mjqs.x = mari - 1;
    mjqs.y = mari;
    mjqs.z = mari + 1;

	
    // --- - z - ---
    marf = (float)(n.z + 0.5);
    mari = (long)(n.z + 0.5);
    t = n.z - marf;
    t2 = t * t;
    t22 = 0.5 * t2;

    // Generate fxqs
    fzqs.x = t22;
    //	fzqs.y = 1 - (t2 + t + 0.5);
    fzqs.y = - t2 + t + 0.5;
    fzqs.z = t22 - t + 0.5;

    // Generate miqs
    mkqs.x = mari - 1;
    mkqs.y = mari;
    mkqs.z = mari + 1;

    // --- Bounds checking
    if (miqs.x < 0) miqs.x = 0;
    if (miqs.y < 0) miqs.y = 0;
    if (miqs.z < 0) miqs.z = 0;
    if (mjqs.x < 0) mjqs.x = 0;	
    if (mjqs.y < 0) mjqs.y = 0;
    if (mjqs.z < 0) mjqs.z = 0;
    if (mkqs.x < 0) mkqs.x = 0;
    if (mkqs.y < 0) mkqs.y = 0;
    if (mkqs.z < 0) mkqs.z = 0;
    // --------------------------------------------------------

    __syncthreads();

    // -- Accumulate Into Segmented Histograms ----------------
    long m_bin;
    long f_bin;
    long j_bin;
    float midx;
    float fidx;
    float mf_1, mf_2;
    float amt;


    // NOTE: j_bin+1 can really fuck us here
    //       if bin is equal to the last bin
    //       in the histogram.
    //
    //  ***: Implement bin bound checking
    //       -AND- go back and do it for
    //       fixed and moving hists as well
    // --- -- - BIN #1 - -- ---
    amt = (1.0/3.0) * (fxqs.y + fyqs.y + fzqs.y);
    mvf = (mkqs.y * mdim.y + mjqs.y) * mdim.x + miqs.y;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;

    // --- -- - BIN #2 - -- ---
    amt = (1.0/3.0) * (fxqs.x);
    mvf = (mkqs.y * mdim.y + mjqs.y) * mdim.x + miqs.x;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;

    // --- -- - BIN #3 - -- ---
    amt = (1.0/3.0) * (fxqs.z);
    mvf = (mkqs.y * mdim.y + mjqs.y) * mdim.x + miqs.z;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;

    // --- -- - BIN #4 - -- ---
    amt = (1.0/3.0) * (fyqs.x);
    mvf = (mkqs.y * mdim.y + mjqs.x) * mdim.x + miqs.y;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;

    // --- -- - BIN #5 - -- ---
    amt = (1.0/3.0) * (fyqs.z);
    mvf = (mkqs.y * mdim.y + mjqs.z) * mdim.x + miqs.y;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;

    // --- -- - BIN #6 - -- ---
    amt = (1.0/3.0) * (fzqs.x);
    mvf = (mkqs.x * mdim.y + mjqs.y) * mdim.x + miqs.y;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;

    // --- -- - BIN #7 - -- ---
    amt = (1.0/3.0) * (fzqs.z);
    mvf = (mkqs.z * mdim.y + mjqs.y) * mdim.x + miqs.y;
    fidx = (fixed[thread_idxg] - f_offset) * f_delta;
    midx = (moving[mvf] - m_offset) * m_delta;
    f_bin = (long)(fidx);
    m_bin = (long)(midx);
    j_bin = (f_bin * m_bins) + m_bin;
    mf_1 = midx - (float)((long)midx);
    mf_2 = 1.0f - mf_1;
    j_hist_seg[j_bin + 0] += mf_1 * amt;
    j_hist_seg[j_bin + 1] += mf_2 * amt;
    //	s_Joint[j_bin + 0] += mf_1 * amt;
    //	s_Joint[j_bin + 1] += mf_2 * amt;
    // --------------------------------------------------------

    __syncthreads();

    // -- Write per Block Histograms out -----------------------
    // NOTE: Because our joint histogram is so big & shared
    //       memory is only 16KB, we only have 1 warp per
    //       thread block.  This means that here we don't
    //       need to merge per warp histograms (there is only 1),
    //       we just need to write it out to global memory.

    //	for (int i=0; i < j_bins; i++)
    //		j_hist_seg[blockIdxInGrid*j_bins + i] = s_Joint[i];

    //	for (long i=0; i < clusters; i++)
    //		if (threadIdx.x + 32*i < j_bins)
    //			j_hist_seg[blockIdxInGrid*j_bins + 32*i] = s_Joint[threadIdx.x + 32*i];

    /*
      for (long i=0; i < clusters; i++)
      if (threadIdx.x + 32*i < j_bins)
      j_hist_seg[blockIdxInGrid*j_bins + 32*i+threadIdx.x] += s_Joint[threadIdx.x + 32*i];
    */

    for (int i=0; i < j_bins; i++)
	j_hist_seg[i] += s_Joint[i];


    // --------------------------------------------------------

    // Done.
    // We now have (num_thread_blocks) partial histograms that
    // need to be merged.  This will be done with another
    // kernel to be ran immediately following the completion
    // of this kernel.

    //NOTE:
    // fv = thread_idxg
    // fi = r.x
    // fj = r.y
    // fk = r.z
}


////////////////////////////////////////////////////////////////////////////////
// Merge Partial/Segmented Histograms
//
//   This kernel is designed to be executed after k_bspline_cuda_MI_a_hist_fix 
//   has genereated many partial histograms (equal to the number of thread-
//   blocks k_bspline_cuda_MI_a_hist_fix() was executed with).  Depending on
//   the image size, this could be as high as hundredes of thousands of
//   partial histograms needing to be merged.
//
//   >> Each thread-block is responsible for a bin number.
//
//   >> A thread-block will use multiple threads to pull down
//      multiple partial histogram bin values in parallel.
//
//   >> Because there are many more partial histograms than threads,
//      the threads in a thread-block will have to iterate through
//      all of the partial histograms using a for-loop.
//
//   >> The # of for-loop iterations is equal to the number of
//      partial histograms divided by the number of threads in a block.
//
//   >> Therefore, this kernel should be launched with:
//
//      -- num_seg_hist % num_threads = 0     (num_seg_hist % blockDim.x = 0)
//      -- num_blocks = num_bins
//
//   >> This means that a search must be executed to find the largest #
//      of threads that can fit within the number of partial histograms
//      we have.  This will exhibit the largest amount of parallelism.
//
////////////////////////////////////////////////////////////////////////////////
__global__ void kernel_bspline_MI_a_hist_fix_merge (
    float *f_hist,
    float *f_hist_seg,
    long num_seg_hist)

{
    extern __shared__ float data[];

    float sum = 0.0f;

    // -- Work through all the sub-histograms ------------------------
    for (long i = threadIdx.x; i < num_seg_hist; i += blockDim.x)
	sum += f_hist_seg[blockIdx.x + i * gridDim.x];

    data[threadIdx.x] = sum;
    // ---------------------------------------------------------------

    __syncthreads();

    // -- Sum all of the thread sums for this bin --------------------
    for (long s = blockDim.x / 2; s > 0; s >>= 1)
    {

	if (threadIdx.x < s)
	    data[threadIdx.x] += data[threadIdx.x + s];

	__syncthreads();
    }
    // ---------------------------------------------------------------


    // -- Write the final bin value to Global ------------------------
    if (threadIdx.x == 0)
	f_hist[blockIdx.x] = data[0];
    // ---------------------------------------------------------------

    // Done.
}



/**
 * Calculates the B-spline score and gradient using CUDA implementation J.
 *
 * @param fixed The fixed volume
 * @param moving The moving volume
 * @param moving_grad The spatial gradient of the moving volume
 * @param bxf Pointer to the B-spline Xform
 * @param parms Pointer to the B-spline parameters
 * @param dev_ptrs Pointer the GPU device pointers
 *
 * @see bspline_cuda_score_j_mse_kernel1()
 * @see CUDA_bspline_mse_2_condense_64_texfetch()
 * @see CUDA_bspline_mse_2_reduce()
 *
 * @author James A. Shackleford
 */
extern "C" void bspline_cuda_j_stage_1 (
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    BSPLINE_Xform* bxf,
    BSPLINE_Parms* parms,
    Dev_Pointers_Bspline* dev_ptrs)
{
    // Reset our "voxels fallen outside" counter
    cudaMemset (dev_ptrs->skipped, 0, dev_ptrs->skipped_size);
    checkCUDAError ("cudaMemset(): dev_ptrs->skipped");
    cudaMemset (dev_ptrs->score, 0, dev_ptrs->score_size);
    checkCUDAError ("cudaMemset(): dev_ptrs->score");

    // Calculate the score and dc_dv
    CUDA_bspline_mse_score_dc_dv (dev_ptrs, bxf, fixed, moving);

    // Prepare for the next kernel
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_g_mse_1");

    // Clear out the condensed dc_dv streams
    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_x");
    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_y");
    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_z");

    /*
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
    */

    // Invoke kernel condense
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    CUDA_bspline_mse_2_condense_64_texfetch (dev_ptrs, bxf->vox_per_rgn, 
	num_tiles);

    /*
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] Condense\n", time);
    */

    // Prepare for the next kernel
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_mse_2_condense()");

    /*
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
    */

    // Clear out the gradient
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    checkCUDAError("cudaMemset(): dev_ptrs->grad");

    // Invoke kernel reduce
    CUDA_bspline_mse_2_reduce (dev_ptrs, bxf->num_knots);

    /*
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] Reduce\n\n", time);
    */

    // Prepare for the next kernel
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_mse_2_condense()");
}



/**
 * Calculates the B-spline score and gradient using CUDA implementation I.
 *
 * @param fixed The fixed volume
 * @param moving The moving volume
 * @param moving_grad The spatial gradient of the moving volume
 * @param bxf Pointer to the B-spline Xform
 * @param parms Pointer to the B-spline parameters
 * @param dev_ptrs Pointer the GPU device pointers
 *
 * @see bspline_cuda_score_g_mse_kernel1()
 * @see CUDA_deinterleave()
 * @see CUDA_pad_64()
 * @see CUDA_bspline_mse_2_condense_64_texfetch()
 * @see CUDA_bspline_mse_2_reduce()
 *
 */
extern "C" void bspline_cuda_i_stage_1 (Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    BSPLINE_Xform* bxf,
    BSPLINE_Parms* parms,
    Dev_Pointers_Bspline* dev_ptrs)
{

    // JAS 10.15.2009
    // TODO: A structure similar to the BSpline_Xform
    //       that uses float3s needs to be used here
    //       to clean up the code.

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

    // Fixed image header
    int3 fix_dim;
    fix_dim.x = fixed->dim[0]; 
    fix_dim.y = fixed->dim[1];
    fix_dim.z = fixed->dim[2];

    float3 fix_origin;		
    fix_origin.x = (float) bxf->img_origin[0];
    fix_origin.y = (float) bxf->img_origin[1];
    fix_origin.z = (float) bxf->img_origin[2];

    float3 fix_spacing;
    fix_spacing.x = (float) bxf->img_spacing[0];
    fix_spacing.y = (float) bxf->img_spacing[1];
    fix_spacing.z = (float) bxf->img_spacing[2];

    // Moving image header
    int3 mov_dim;		
    mov_dim.x = moving->dim[0]; 
    mov_dim.y = moving->dim[1];
    mov_dim.z = moving->dim[2];

    float3 mov_origin;
    mov_origin.x = (float) moving->offset[0];
    mov_origin.y = (float) moving->offset[1];
    mov_origin.z = (float) moving->offset[2];

    float3 mov_spacing;
    mov_spacing.x = (float) moving->pix_spacing[0];
    mov_spacing.y = (float) moving->pix_spacing[1];
    mov_spacing.z = (float) moving->pix_spacing[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];


    // JAS 10.15.2009
    // TODO: The following blocked off section needs
    //       to be turned into its own function.
    // ------------------------------------------------
    // ------------------------------------------------

    // --- INITIALIZE GRID ---
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 128;
    int num_threads = fixed->npix;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    int smemSize = 12 * sizeof(float) * threads_per_block;


    // -----
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // -----


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable bspline_cuda_score_j_mse_kernel1() configuration!\n");
	exit(0);
    } else {
#if defined (commentout)
	printf ("Grid [%i,%i], %d threads_per_block.\n", 
	    Grid_x, Grid_y, threads_per_block);
#endif
    }

    dim3 dimGrid1(Grid_x, Grid_y, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);
    // ----------------------


    // --- BEGIN KERNEL EXECUTION ---
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord (start, 0);	

    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_x");

    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_y");

    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_z");

    cudaMemset(dev_ptrs->skipped, 0, dev_ptrs->skipped_size);
    checkCUDAError("cudaMemset(): dev_ptrs->skipped");

    int tile_padding = 64 - ((vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z) % 64);

    bspline_cuda_score_j_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>(
	dev_ptrs->dc_dv_x,	// Addr of dc_dv_x on GPU
	dev_ptrs->dc_dv_y,	// Addr of dc_dv_y on GPU
	dev_ptrs->dc_dv_z,	// Addr of dc_dv_z on GPU
	dev_ptrs->score,	// Addr of score on GPU
	dev_ptrs->coeff,	// Addr of coeff on GPU
	dev_ptrs->fixed_image,	// Addr of fixed_image on GPU
	dev_ptrs->moving_image,	// Addr of moving_image on GPU
	dev_ptrs->moving_grad,  // Addr of moving_grad on GPU
	fix_dim,                // Size of fixed image (vox)
	fix_origin,             // Origin of fixed image (mm)
	fix_spacing,            // Spacing of fixed image (mm)
	mov_dim,                // Size of moving image (vox)
	mov_origin,             // Origin of moving image (mm)
	mov_spacing,            // Spacing of moving image (mm)
	roi_dim,		// Region of Intrest Dimenions
	roi_offset,		// Region of Intrest Offset
	vox_per_rgn,		// Voxels per Region
	rdims,			// 
	cdims,
	tile_padding,
	dev_ptrs->skipped);

    cudaEventRecord (stop, 0);	
    cudaEventSynchronize (stop);

    cudaEventElapsedTime (&time, start, stop);

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    printf("\n[%f ms] MSE & dc_dv\n", time);
    // ------------------------------

    // END: Needs to be turned into its own function.
    // ----------------------------------------------------------
    // ----------------------------------------------------------



    // Prepare for the next kernel
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_g_mse_1");

    // Clear out the condensed dc_dv streams
    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_z");


    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);

    // Invoke kernel condense
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    CUDA_bspline_mse_2_condense_64 (dev_ptrs, bxf->vox_per_rgn, num_tiles);

    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] Condense\n", time);

    // Prepare for the next kernel
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_mse_2_condense()");

    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);

    // Clear out the gradient
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    checkCUDAError("cudaMemset(): dev_ptrs->grad");

    // Invoke kernel reduce
    CUDA_bspline_mse_2_reduce (dev_ptrs, bxf->num_knots);

    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] Reduce\n\n", time);

    // Prepare for the next kernel
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_mse_2_condense()");

    // END OF FUNCTION

    // ORIGIONAL i STARTS HERE.
    // Replaced with a version of j that performs on the fly
    // calculations instead of using a LUT for B-Spline coefficients.
    //
    // The following is maintained for reference purposes.
    /*
    // --- INITIALIZE LOCAL VARIABLES ---------------------------

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
    // ----------------------------------------------------------


    // --- INITIALIZE GRID -------------------------------------
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 128;
    int num_threads = fixed->npix;
    //	int num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    int smemSize = 12 * sizeof(float) * threads_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
    if (num_blocks % i == 0)
    {
    Grid_x = i;
    Grid_y = num_blocks / Grid_x;
    break;
    }
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
    // If this happens we should consider falling back to a
    // CPU implementation, using a different CUDA algorithm,
    // or padding the input dc_dv stream to work with this
    // CUDA algorithm.
    printf("\n[ERROR] Unable to find suitable bspline_cuda_score_g_mse_kernel1() configuration!\n");
    exit(0);
    } else {
#if defined (commentout)
	printf ("Grid [%i,%i], %d threads_per_block.\n", 
	    Grid_x, Grid_y, threads_per_block);
#endif
    }

    dim3 dimGrid1(Grid_x, Grid_y, 1);


    //	dim3 dimGrid1(num_blocks / 128, 128, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    // (a.k.a: bspline_cuda_score_g_mse_kernel1)

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord (start, 0);	


    // For now we are using the legacy g_mse_kernel1
    // later to be replaced with h_mse_kernel1
    bspline_cuda_score_g_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>(
    dev_ptrs->dc_dv,	// Addr of dc_dv on GPU
    dev_ptrs->score,	// Addr of score on GPU
    dev_ptrs->coeff,	// Addr of coeff on GPU
    dev_ptrs->fixed_image,	// Addr of fixed_image on GPU
    dev_ptrs->moving_image,	// Addr of moving_image on GPU
    dev_ptrs->moving_grad,  // Addr of moving_grad on GPU
    volume_dim,		// Volume Dimensions
    img_origin,		// Origin
    img_spacing,		// Voxel Spacing
    img_offset,		// Image Offset
    roi_offset,		// Region of Intrest Offset
    roi_dim,		// Region of Intrest Dimenions
    vox_per_rgn,		// Voxels per Region
    pix_spacing,		// Pixel Spacing
    rdims,			// 
    cdims);			// 

    cudaEventRecord (stop, 0);	
    cudaEventSynchronize (stop);

    cudaEventElapsedTime (&time, start, stop);

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    printf("\n[%f ms] G Part 1\n", time);
    // ----------------------------------------------------------



    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_g_mse_1");
    // ----------------------------------------------------------
	
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&& PART 2 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


    // !!! START TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventCreate(&start);                                    //!!
    cudaEventCreate(&stop);                                     //!!
    cudaEventRecord (start, 0);                                 //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // ----------------------------------------------------------
    // * Glue Code 1
    //    [GPU] Generate 3 seperate Row-Major dc_dv volumes
    //          One for X, one for Y, and one for Z
    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_x");

    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_y");

    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_z");

    CUDA_deinterleave(dev_ptrs->dc_dv_size/sizeof(float),
    dev_ptrs->dc_dv,
    dev_ptrs->dc_dv_x,
    dev_ptrs->dc_dv_y,
    dev_ptrs->dc_dv_z);

    // Release dc_dv on the card so we have enough memory
    // (We will have to re-allocate dc_dv before we return)
    //	cudaUnbindTexture (tex_dc_dv);
    //	cudaFree( dev_ptrs->dc_dv );
    // ----------------------------------------------------------

    // !!! STOP TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventRecord (stop, 0);                                  //!!
    cudaEventSynchronize (stop);                                //!!
    cudaEventElapsedTime (&time, start, stop);                  //!!
    cudaEventDestroy (start);                                   //!!
    cudaEventDestroy (stop);                                    //!!
    printf("[%f ms] Deinterleaving\n", time);                   //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    // !!! START TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventCreate(&start);                                    //!!
    cudaEventCreate(&stop);                                     //!!
    cudaEventRecord (start, 0);                                 //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	
    // ----------------------------------------------------------
    // * Glue Code 2
    //    [GPU] Convert the 3 deinterleaved row-major
    //          data streams into 3 32-byte aligned
    //          tiled streams.
    CUDA_pad_64(&dev_ptrs->dc_dv_x,
    fixed->dim,
    bxf->vox_per_rgn);

    CUDA_pad_64(&dev_ptrs->dc_dv_y,
    fixed->dim,
    bxf->vox_per_rgn);

    CUDA_pad_64(&dev_ptrs->dc_dv_z,
    fixed->dim,
    bxf->vox_per_rgn);
    // ----------------------------------------------------------

    // !!! STOP TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventRecord (stop, 0);                                  //!!
    cudaEventSynchronize (stop);                                //!!
    cudaEventElapsedTime (&time, start, stop);                  //!!
    cudaEventDestroy (start);                                   //!!
    cudaEventDestroy (stop);                                    //!!
    printf("[%f ms] Data Padding\n", time);                     //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // ----------------------------------------------------------
    // * Setup 3
    //     Clear out the condensed dc_dv streams
	
    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_z");
    // ----------------------------------------------------------


    // !!! START TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventCreate(&start);                                    //!!
    cudaEventCreate(&stop);                                     //!!
    cudaEventRecord (start, 0);                                 //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    // --- INVOKE KERNEL CONDENSE -------------------------------
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    CUDA_bspline_mse_2_condense_64(dev_ptrs, bxf->vox_per_rgn, num_tiles);
    //	CPU_bspline_mse_2_condense(dev_ptrs, bxf->vox_per_rgn, bxf->cdims, bxf->rdims, num_tiles);
    // ----------------------------------------------------------

    // !!! STOP TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventRecord (stop, 0);                                  //!!
    cudaEventSynchronize (stop);                                //!!
    cudaEventElapsedTime (&time, start, stop);                  //!!
    cudaEventDestroy (start);                                   //!!
    cudaEventDestroy (stop);                                    //!!
    printf("[%f ms] Condense\n", time);                         //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_mse_2_condense()");
    // ----------------------------------------------------------

    // !!! START TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventCreate(&start);                                    //!!
    cudaEventCreate(&stop);                                     //!!
    cudaEventRecord (start, 0);                                 //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    // --- INVOKE KERNEL CONDENSE -------------------------------
    CUDA_bspline_mse_2_reduce(dev_ptrs, bxf->num_knots);
    //	CPU_bspline_mse_2_reduce(dev_ptrs, bxf->num_knots);
    // ----------------------------------------------------------


    // !!! STOP TIMING THE KERNEL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cudaEventRecord (stop, 0);                                  //!!
    cudaEventSynchronize (stop);                                //!!
    cudaEventElapsedTime (&time, start, stop);                  //!!
    cudaEventDestroy (start);                                   //!!
    cudaEventDestroy (stop);                                    //!!
    printf("[%f ms] Reduce\n\n", time);                         //!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_bspline_mse_2_condense()");
    // ----------------------------------------------------------


    // --- PUT dc_dv BACK THE WAY WE FOUND IT -------------------
    // This is some disabled LOW-MEM code.  We don't need
    // to de-allocate and re-allocate dc_dv, but we can if
    // we are in dire need for more memory.  The re-allocation
    // process is a little slow, so we waste a little memory
    // here in a trade off for speed.

    // Re-Allocate dev_ptrs->dc_dv
    //	cudaMalloc((void**)&dev_ptrs->dc_dv, dev_ptrs->dc_dv_size);
    //	cudaMemset(dev_ptrs->dc_dv, 0, dev_ptrs->dc_dv_size);
    //	cudaBindTexture(0, tex_dc_dv, dev_ptrs->dc_dv, dev_ptrs->dc_dv_size);
    // ----------------------------------------------------------

    */
    // END OF REFERENCE CODE

}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// STUB: bspline_cuda_j_stage_2()
//
// KERNELS INVOKED:
//   sum_reduction_kernel()
//   sum_reduction_last_step_kernel()
//   bspline_cuda_update_grad_kernel()
//   bspline_cuda_compute_grad_mean_kernel()
//   sum_reduction_last_step_kernel()
//   bspline_cuda_compute_grad_norm_kernel
//   sum_reduction_last_step_kernel()
//
// bspline_cuda_final_steps_f()
////////////////////////////////////////////////////////////////////////////////
extern "C" void bspline_cuda_j_stage_2 (
    BSPLINE_Parms* parms, 
    BSPLINE_Xform* bxf,
    Volume* fixed,
    int*   vox_per_rgn,
    int*   volume_dim,
    float* host_score,
    float* host_grad,
    float* host_grad_mean,
    float* host_grad_norm,
    Dev_Pointers_Bspline* dev_ptrs,
    int *num_vox)
{

    // --- INITIALIZE GRID --------------------------------------
    int Grid_x = 0;
    int Grid_y = 0;
    int num_elems = volume_dim[0] * volume_dim[1] * volume_dim[2];
    //	int num_blocks = (int)ceil(num_elems / 512.0);
    int num_blocks = (num_elems + 511) / 512;
	
    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    int i;
    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****

    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable sum_reduction_kernel() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting sum_reduction_kernel() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    dim3 dimBlock(128, 2, 2);
    int smemSize = 512 * sizeof(float);
    // ----------------------------------------------------------

#if defined (commentout)
    /* Compute score on cpu for debugging */
    {
	int i;
	float *cpu_score = (float*) malloc (dev_ptrs->score_size);
	int num_ele = dev_ptrs->score_size / sizeof (float);
	double sse = 0.0;
	FILE *fp;

	cudaMemcpy (cpu_score, dev_ptrs->score, dev_ptrs->score_size, 
	    cudaMemcpyDeviceToHost);
	for (i = 0; i < num_ele; i++) {
	    sse += (double) cpu_score[i];
	}
	sse /= 128480.;
	printf ("CPU computed score as %f\n", sse);
	
	fp = fopen ("gpu_score.txt", "wb");
	for (i = 0; i < num_ele; i++) {
	    fprintf (fp, "%f\n", cpu_score[i]);
	}
	fclose (fp);
    }
#endif

    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
	dev_ptrs->score,
	dev_ptrs->score,
	num_elems
    );
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_sum_reduction()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
	dev_ptrs->score,
	dev_ptrs->score,
	num_elems
    );
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_sum_reduction_last_step()");
    // ----------------------------------------------------------


    // --- RETREIVE THE SCORE FROM GPU --------------------------
    cudaMemcpy(host_score, dev_ptrs->score,  sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDAError("Failed to copy score from GPU to host");
    // ----------------------------------------------------------


    //	for (i = 1; i < (dev_ptrs->skipped_size / sizeof(int)); i++)
    //		skipped[0] += skipped[i];

    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
	dev_ptrs->skipped,
	dev_ptrs->skipped,
	num_elems
    );
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_sum_reduction()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
	dev_ptrs->skipped,
	dev_ptrs->skipped,
	num_elems
    );
    // ----------------------------------------------------------

    float skipped;
    cudaMemcpy(&skipped, dev_ptrs->skipped, sizeof(float), cudaMemcpyDeviceToHost);

    *num_vox = (volume_dim[0] * volume_dim[1] * volume_dim[2]) - skipped;

    *host_score = *host_score / *num_vox;

    /////////////////////////////////////////////////////////////
    /////////////////////// CALCULATE ///////////////////////////
    ////////////// GRAD, GRAD NORM *AND* GRAD MEAN //////////////
    /////////////////////////////////////////////////////////////


    // --- RE-INITIALIZE GRID -----------------------------------
    Grid_x = 0;
    Grid_y = 0;
    num_elems = bxf->num_coeff;
    //	num_blocks = (int)ceil(num_elems / 512.0);
    num_blocks = (num_elems + 511) / 512;
	
    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****

    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable sum_reduction_kernel() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting sum_reduction_kernel() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid2(Grid_x, Grid_y, 1);
    dim3 dimBlock2(128, 2, 2);
    smemSize = 512 * sizeof(float);
    // ----------------------------------------------------------
	

    // --- BEGIN KERNEL EXECUTION -------------------------------
    bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
	dev_ptrs->grad,
	*num_vox,
	num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] bspline_cuda_update_grad_kernel");
    // ----------------------------------------------------------


    // --- RETREIVE THE GRAD FROM GPU ---------------------------
    cudaMemcpy(host_grad, dev_ptrs->grad, sizeof(float) * bxf->num_coeff, cudaMemcpyDeviceToHost);
    checkCUDAError("Failed to copy dev_ptrs->grad to CPU");
    // ----------------------------------------------------------

    // The following is unnecessary since report_score()
    // calculates the grad mean & norm from grad[] anyway.
    /*
    // --- BEGIN KERNEL EXECUTION -------------------------------
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
    dev_ptrs->grad,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] bspline_cuda_grad_mean_kernel()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
    dev_ptrs->grad_temp,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_sum_reduction_last_step()");
    // ----------------------------------------------------------


    // --- RETREIVE THE GRAD MEAN FROM GPU ----------------------
    cudaMemcpy(host_grad_mean, dev_ptrs->grad_temp, sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDAError("Failed to copy grad_mean from GPU to host");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
    dev_ptrs->grad,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] bspline_cuda_compute_grad_norm_kernel()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
    dev_ptrs->grad_temp,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_sum_reduction_last_step()");
    // ----------------------------------------------------------


    // --- RETREIVE THE GRAD NORM FROM GPU ----------------------
    cudaMemcpy(host_grad_norm, dev_ptrs->grad_temp, sizeof(float), cudaMemcpyDeviceToHost);
    checkCUDAError("Failed to copy grad_norm from GPU to host");
    // ----------------------------------------------------------
    */
}



/**
 * Deinterleaves 3-interleaved data and generates three deinterleaved arrays.
 *
 * @param num_values Deprecated
 * @param input Pointer to memory containing interleaved data
 * @param out_x Pointer to memory containing the deinterleaved x-values
 * @param out_y Pointer to memory containing the deinterleaved y-values
 * @param out_z Pointer to memory containing the deinterleaved z-values
 *
 * @author James A. Shackleford
 */
__global__ void kernel_deinterleave(
    int num_values,
    float* input,
    float* out_x,
    float* out_y,
    float* out_z)
{
    // Shared memory is allocated on a per block basis.
    // (Allocate (2*96*sizeof(float)) memory when calling the kernel.)
    extern __shared__ float shared_memory[]; 

    float* sdata = (float*)shared_memory;		// float sdata[96];
    float* sdata_x = (float*)&sdata[96];		// float sdata_x[32];
    float* sdata_y = (float*)&sdata_x[32];		// float sdata_y[32];
    float* sdata_z = (float*)&sdata_y[32];		// float sdata_z[32];


    // Total shared memory allocation per block: 2*96*sizeof(float)
	

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // The total number of threads in each thread block.
    int threadsPerBlock  = 96;

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // Used for determining Warp Number
    int warpNumber = threadIdxInBlock / 32;

    ///////////////////////////////////////
    // We have 3 warps (96 threads).
    // 96 threads is enough to pull down:
    //   -- 32 X values
    //   -- 32 Y values
    //   -- 32 Z values
    ////////////////////////////

    // First, we will pull these 96 values into shared memory.
    // At this point they will still be interleaved in shared memory
    sdata[threadIdxInBlock] = input[threadIdxInGrid];
	
    __syncthreads();


    // Second, each warp will diverge.  (This is okay because we
    // are diverging along warp boundaries.)  Each warp will be
    // responsible for deinterleaving 1/3 of the values stored in
    // shared memory and copying them to one of 3 other areas
    // in shared memory.
    //   -- Warp 0 will grab the X values
    //   -- Warp 1 will grab the Y values
    //   -- Warp 2 will grab the Z values
    switch (warpNumber)
    {
    case 0:
	sdata_x[threadIdxInBlock] = sdata[3*threadIdxInBlock];
	break;

    case 1:
	sdata_y[threadIdxInBlock - 32] = sdata[3*threadIdxInBlock - 95];
	break;
		
    case 2:
	sdata_z[threadIdxInBlock - 64] = sdata[3*threadIdxInBlock - 190];
	break;
    }

    __syncthreads();


    // Finally, each warp is now responsible for one of the coalesced
    // X, Y, or Z streams in shared memory.  The job is to now
    // move these contigious elements into global memory.
    switch (warpNumber)
    {
    case 0:
	out_x[threadIdxInBlock + 32*blockIdxInGrid] = sdata_x[threadIdxInBlock];
	break;

    case 1:
	out_y[(threadIdxInBlock - 32) + 32*blockIdxInGrid] = sdata_y[threadIdxInBlock - 32];
	break;
		
    case 2:
	out_z[(threadIdxInBlock - 64) + 32*blockIdxInGrid] = sdata_z[threadIdxInBlock - 64];
	break;
    }

}



/**
 * This kernel converts a row-major data stream into a 32-byte aligned
 * tile-major stream.
 *
 * @warning Invoke with as many threads as there are elements in the row-major data.
 *
 * @param input Pointer to the input row-major data
 * @param output Pointer to the output tiled data
 * @param vol_dim Dimensions of the row-major data volume
 * @param tile_dim Desired dimensions of the tiles
 *
 * @author James A. Shackleford
 */
__global__ void kernel_row_to_tile_major(
    float* input,
    float* output,
    int3 vol_dim,
    int3 tile_dim)
{
    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    if (threadIdxInGrid >= (vol_dim.x * vol_dim.y * vol_dim.z))
	return;

    // How many tiles do we need in the x, y, and z directions
    // in order to accommodate the volume?
    int3 num_tiles;
    num_tiles.x = (vol_dim.x+tile_dim.x-1) / tile_dim.x;
    num_tiles.y = (vol_dim.y+tile_dim.y-1) / tile_dim.y;
    num_tiles.z = (vol_dim.z+tile_dim.z-1) / tile_dim.z;

    // Setup shared memory
    extern __shared__ float sdata[]; 

    // We must first calculate where each tile will start in
    // memory.  This is the same as saying the start of each tile
    // in linear memory.  In linear memory, tiles are separated
    // by buffers of unused data so that each tile starts at a
    // 32-byte boundary.

    // The first step will be to find by how much we must pad
    // each tile so that it is divisible evenly by 32.
    int tile_padding = 32 - ((tile_dim.x * tile_dim.y * tile_dim.z) % 32);

    // Now each thread maps to one voxel in the row-major volume.
    // We will use the threadIdx to figure out which tile
    // the voxel we are operating on maps into in the tile-major
    // volume.

    // But first, we must find the [x,y,z] coordinates of the
    // voxel we are operating on based on the threadIdxInGrid
    // and the volume dimensions.
    int3 vox_coord;
    vox_coord.x = threadIdxInGrid % vol_dim.x;
    vox_coord.y = ((threadIdxInGrid - vox_coord.x) / vol_dim.x) % vol_dim.y;
    vox_coord.z = ((((threadIdxInGrid - vox_coord.x) / vol_dim.x) / vol_dim.y) % vol_dim.z);

    // ...and now we can find the voxel's destination tile
    // in the tile-major volume.
    int4 dest_tile;
    dest_tile.x = vox_coord.x / tile_dim.x;
    dest_tile.y = vox_coord.y / tile_dim.y;
    dest_tile.z = vox_coord.z / tile_dim.z;
	
    // ...and based on the destination tile [x,y,z] coordinates
    // we find the *TILE's* absolute row-major offset (and store it
    // into dest_tile.w).
    dest_tile.w = num_tiles.x*num_tiles.y*dest_tile.z + num_tiles.x*dest_tile.y + dest_tile.x;

    // Multiplying the destination tile number by the tile_padding
    // tells us our padding offset for where the destination tile lives
    // in linear memory.
    int linear_mem_offset_pad = tile_padding * dest_tile.w;

    // We can also find the linear memory offset of the tile
    // due to all of the voxels contained within the tiles preceeding
    // it.
    int linear_mem_offset_tile = (tile_dim.x*tile_dim.y*tile_dim.z) * dest_tile.w;

    // Now we can find the effective offset into linear
    // memory for our tile.
    int linear_mem_offset = linear_mem_offset_tile + linear_mem_offset_pad;

    // Now that we have the linear offset of where our
    // tile starts in linear memory (which will be on
    // a 32-byte boundary btw), we can now focus on
    // what the destination coordinates of our voxel
    // will be within that tile.
	
    // We will call the voxel coordinates within the
    // tile dest_coords.  The final location of our
    // voxel in linear memory will be:
    // linear_mem_offset + dest_coord.w
    int4 dest_coord;
    dest_coord.x = vox_coord.x - (dest_tile.x * tile_dim.x);
    dest_coord.y = vox_coord.y - (dest_tile.y * tile_dim.y);
    dest_coord.z = vox_coord.z - (dest_tile.z * tile_dim.z);
    dest_coord.w = tile_dim.x*tile_dim.y*dest_coord.z + tile_dim.x*dest_coord.y + dest_coord.x;
	
    // We now, FINALLY, know where our row-major voxel
    // maps to in linear memory for our 32-byte aligned
    // tile-major volume!  \(^_^)/ YATTA! \(^_^)/
    int linear_mem_idx = linear_mem_offset + dest_coord.w;


    // Lets move it!
    //	output[linear_mem_idx] = (float)threadIdxInGrid;	// Output Check
    output[linear_mem_idx] = input[threadIdxInGrid];
	

    // Fin.
}


/**
 * This kernel pads tiled data so that each tile is aligned to 64 byte boundaries
 *
 * @param input Pointer to tiled data
 * @param output Pointer to padded tiled data
 * @param vol_dim Dimensions of input data volume
 * @param tile_dim Dimension of input data volume's tiles
 *
 * @author James A. Shackleford
 */
__global__ void kernel_pad_64(
    float* input,
    float* output,
    int3 vol_dim,
    int3 tile_dim)
{
    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    if (threadIdxInGrid >= (vol_dim.x * vol_dim.y * vol_dim.z))
	return;

    int num_elements = tile_dim.x * tile_dim.y * tile_dim.z;

    // "Which tile am I handling," wondered the warp.
    int tile_id = threadIdxInGrid / num_elements;
	
    // "Hmm... a pad," said the thread with intrigue.
    int tile_padding = 64 - (num_elements % 64);

    // "We'll need an offset as well," he said.
    int offset = tile_id * (tile_padding + num_elements);

    int idx = threadIdxInGrid - (tile_id * num_elements);

    // This story sucks... let's just get this over with.
    output[offset + idx] = input[threadIdxInGrid];
	
}



/**
 * This kernel pads tiled data so that each tile is aligned to 32 byte boundaries
 *
 * @param input Pointer to tiled data
 * @param output Pointer to padded tiled data
 * @param vol_dim Dimensions of input data volume
 * @param tile_dim Dimension of input data volume's tiles
 *
 * @author James A. Shackleford
 */
__global__ void kernel_pad(
    float* input,
    float* output,
    int3 vol_dim,
    int3 tile_dim)
{
    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    if (threadIdxInGrid >= (vol_dim.x * vol_dim.y * vol_dim.z))
	return;

    int num_elements = tile_dim.x * tile_dim.y * tile_dim.z;

    int tile_id = threadIdxInGrid / num_elements;
	
    int tile_padding = 32 - (num_elements % 32);

    int offset = tile_id * (tile_padding + num_elements);

    int idx = threadIdxInGrid - (tile_id * num_elements);

    output[offset + idx] = input[threadIdxInGrid];
	
}



/**
 * This kernel partially computes the gradient by generating condensed dc_dv values.
 *
 * @warning It is required that input data tiles be aligned to 64 byte boundaries.
 *
 * @see CUDA_pad_64()
 * @see kernel_pad_64()
 *
 * @param cond_x Pointer to condensed dc_dv x-values
 * @param cond_y Pointer to condensed dc_dv y-values
 * @param cond_z Pointer to condensed dc_dv z-values
 * @param dc_dv_x Pointer to dc_dv x-values
 * @param dc_dv_y Pointer to dc_dv y-values
 * @param dc_dv_z Pointer to dc_dv z-values
 * @param LUT_Tile_Offsets Pointer to offset lookup table
 * @param LUT_Knot Pointer to linear knot indices
 * @param pad Amount of tile padding, in bytes
 * @param tile_dim Dimensions of input volume tiles
 * @param one_over_six The value 1/6
 *
 * @author: James A. Shackleford
 */
__global__ void kernel_bspline_mse_2_condense_64_texfetch (
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
    float one_over_six)	// Input : Precomputed since GPU division is slow
{
    int tileOffset;
    int voxel_cluster;
    int voxel_idx;
    float3 voxel_val;
    int3 voxel_loc;
    int4 tile_pos;
    float A,B,C;


    // -- Setup Thread Attributes -----------------------------
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    // --------------------------------------------------------


    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: 9*64*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer_x = (float*)sdata;			// sBuffer_x[64]
    float* sBuffer_y = (float*)&sBuffer_x[64];		// sBuffer_y[64]
    float* sBuffer_z = (float*)&sBuffer_y[64];		// sBuffer_z[64]
    float* sBuffer_redux_x = (float*)&sBuffer_z[64];	// sBuffer_redux_x[64]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[64];	// sBuffer_redux_y[64]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[64];	// sBuffer_redux_z[64]
    float* sBuffer_redux_x2 = (float*)&sBuffer_redux_z[64];	// sBuffer_redux_x2[64]
    float* sBuffer_redux_y2 = (float*)&sBuffer_redux_x2[64];// sBuffer_redux_y2[64]
    float* sBuffer_redux_z2 = (float*)&sBuffer_redux_y2[64];// sBuffer_redux_z2[64]
    // --------------------------------------------------------


    // Clear Shared Memory!!
    sBuffer_x[threadIdx.x] = 0.0f;
    sBuffer_y[threadIdx.x] = 0.0f;
    sBuffer_z[threadIdx.x] = 0.0f;


    // First, get the offset of where our tile starts in memory.
    tileOffset = LUT_Tile_Offsets[blockIdxInGrid];

    // Main Loop for Warp Work
    // (Here we condense a tile into 64x3 floats)
    for (voxel_cluster=0; voxel_cluster < tile_dim.w; voxel_cluster+=64)
    {

	// ----------------------------------------------------------
	//                  STAGE 1 IN POWERPOINT
	// ----------------------------------------------------------
	// Second, we pulldown the current voxel cluster.
	// Each thread in the warp pulls down 1 voxel (3 values)
	// ----------------------------------------------------------
	voxel_val.x = dc_dv_x[tileOffset + voxel_cluster + threadIdx.x];
	voxel_val.y = dc_dv_y[tileOffset + voxel_cluster + threadIdx.x];
	voxel_val.z = dc_dv_z[tileOffset + voxel_cluster + threadIdx.x];
	// ----------------------------------------------------------

	// Third, find the [x,y,z] location within the current tile
	// for the voxel this thread is processing.
	voxel_idx = (voxel_cluster + threadIdx.x);
	voxel_loc.z = voxel_idx / (tile_dim.x * tile_dim.y);
	voxel_loc.y = (voxel_idx - (voxel_loc.z * tile_dim.x * tile_dim.y)) / tile_dim.x;
	voxel_loc.x = voxel_idx - voxel_loc.z * tile_dim.x * tile_dim.y - (voxel_loc.y * tile_dim.x);

	// Fourth, we will perform all 64x3 calculations on the current voxel cluster.
	// (Every thead in the warp will be doing this at the same time for its voxel)

	tile_pos.w = 0;	// Current tile position within [0,63]

	for (tile_pos.z = 0; tile_pos.z < 4; tile_pos.z++)
	{
	    C = TEX_REF(LUT_Bspline_z, tile_pos.z * tile_dim.z + voxel_loc.z);
	    for (tile_pos.y = 0; tile_pos.y < 4; tile_pos.y++)
	    {
		B = C * TEX_REF(LUT_Bspline_y, tile_pos.y * tile_dim.y + voxel_loc.y);
		tile_pos.x = 0;

		// #### FIRST HALF ####

		// ---------------------------------------------------------------------------------
		// Do the 1st two x-positions out of four using our two
		// blocks of shared memory for reduction

		// Calculate the b-spline multiplier for this voxel @ this tile
		// position relative to a given control knot.
		// ---------------------------------------------------------------------------------
		A = B * TEX_REF(LUT_Bspline_x, tile_pos.x * tile_dim.x + voxel_loc.x);

		// Perform the multiplication and store to redux shared memory
		sBuffer_redux_x[threadIdx.x] = voxel_val.x * A;
		sBuffer_redux_y[threadIdx.x] = voxel_val.y * A;
		sBuffer_redux_z[threadIdx.x] = voxel_val.z * A;
		tile_pos.x++;

		// Calculate the b-spline multiplier for this voxel @ the next tile
		// position relative to a given control knot.
		A = B * TEX_REF(LUT_Bspline_x, tile_pos.x * tile_dim.x + voxel_loc.x);

		// Perform the multiplication and store to redux shared memory
		// for the second position
		sBuffer_redux_x2[threadIdx.x] = voxel_val.x * A;
		sBuffer_redux_y2[threadIdx.x] = voxel_val.y * A;
		sBuffer_redux_z2[threadIdx.x] = voxel_val.z * A;
		__syncthreads();
		// ---------------------------------------------------------------------------------


		// ---------------------------------------------------------------------------------
		// All 64 dc_dv values in the current cluster have been processed
		// for the current 2 tile positions (out of 64 total tile positions).
				
		// We now perform a sum reduction on these 64 dc_dv values to
		// condense the data down to one value.
		// ---------------------------------------------------------------------------------
		if (threadIdx.x < 32)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 32];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 32];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 32];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 32];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 32];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 32];
		}
		__syncthreads();

		if (threadIdx.x < 16)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 16];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 16];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 16];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 16];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 16];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 16];
		}
		__syncthreads();

		if (threadIdx.x < 8)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 8];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 8];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 8];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 8];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 8];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 8];
		}
		__syncthreads();

		if (threadIdx.x < 4)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 4];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 4];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 4];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 4];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 4];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 4];
		}
		__syncthreads();

		if (threadIdx.x < 2)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 2];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 2];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 2];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 2];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 2];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 2];
		}
		__syncthreads();

		if (threadIdx.x < 1)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 1];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 1];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 1];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 1];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 1];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 1];
		}
		__syncthreads();
		// ---------------------------------------------------------------------------------



		// ---------------------------------------------------------------------------------
		// We then accumulate this single condensed value into the element of
		// shared memory that correlates to the current tile position.
		// ---------------------------------------------------------------------------------
		if (threadIdx.x == 0)
		{
		    sBuffer_x[tile_pos.w] += sBuffer_redux_x[0];
		    sBuffer_y[tile_pos.w] += sBuffer_redux_y[0];
		    sBuffer_z[tile_pos.w] += sBuffer_redux_z[0];
		    tile_pos.w++;

		    sBuffer_x[tile_pos.w] += sBuffer_redux_x2[0];
		    sBuffer_y[tile_pos.w] += sBuffer_redux_y2[0];
		    sBuffer_z[tile_pos.w] += sBuffer_redux_z2[0];
		    tile_pos.w++;
		}
		__syncthreads();
		// ---------------------------------------------------------------------------------


		// #### SECOND HALF ####

		// ---------------------------------------------------------------------------------
		// Do the 2nd two x-positions out of four using our two
		// blocks of shared memory for reduction
		// ---------------------------------------------------------------------------------
		tile_pos.x++;
		A = B * TEX_REF(LUT_Bspline_x, tile_pos.x * tile_dim.x + voxel_loc.x);

		// Perform the multiplication and store to redux shared memory
		sBuffer_redux_x[threadIdx.x] = voxel_val.x * A;
		sBuffer_redux_y[threadIdx.x] = voxel_val.y * A;
		sBuffer_redux_z[threadIdx.x] = voxel_val.z * A;
		tile_pos.x++;

		// Calculate the b-spline multiplier for this voxel @ the next tile
		// position relative to a given control knot.
		A = B * TEX_REF(LUT_Bspline_x, tile_pos.x * tile_dim.x + voxel_loc.x);

		// Perform the multiplication and store to redux shared memory
		// for the second position
		sBuffer_redux_x2[threadIdx.x] = voxel_val.x * A;
		sBuffer_redux_y2[threadIdx.x] = voxel_val.y * A;
		sBuffer_redux_z2[threadIdx.x] = voxel_val.z * A;
		__syncthreads();
		// ---------------------------------------------------------------------------------


					
		// ---------------------------------------------------------------------------------
		// All 64 dc_dv values in the current cluster have been processed
		// for the current 2 tile positions (out of 64 total tile positions).
		//
		// We now perform a sum reduction on these 64 dc_dv values to
		// condense the data down to one value.
		// ---------------------------------------------------------------------------------
		if (threadIdx.x < 32)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 32];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 32];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 32];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 32];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 32];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 32];
		}
		__syncthreads();

		if (threadIdx.x < 16)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 16];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 16];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 16];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 16];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 16];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 16];
		}
		__syncthreads();

		if (threadIdx.x < 8)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 8];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 8];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 8];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 8];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 8];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 8];
		}
		__syncthreads();

		if (threadIdx.x < 4)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 4];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 4];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 4];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 4];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 4];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 4];
		}
		__syncthreads();

		if (threadIdx.x < 2)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 2];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 2];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 2];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 2];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 2];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 2];
		}
		__syncthreads();

		if (threadIdx.x < 1)
		{
		    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + 1];
		    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + 1];
		    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + 1];
		    sBuffer_redux_x2[threadIdx.x] += sBuffer_redux_x2[threadIdx.x + 1];
		    sBuffer_redux_y2[threadIdx.x] += sBuffer_redux_y2[threadIdx.x + 1];
		    sBuffer_redux_z2[threadIdx.x] += sBuffer_redux_z2[threadIdx.x + 1];
		}
		__syncthreads();
		// ---------------------------------------------------------------------------------



		// ---------------------------------------------------------------------------------
		// We then accumulate this single condensed value into the element of
		// shared memory that correlates to the current tile position.
		// ---------------------------------------------------------------------------------
		if (threadIdx.x == 0)
		{
		    sBuffer_x[tile_pos.w] += sBuffer_redux_x[0];
		    sBuffer_y[tile_pos.w] += sBuffer_redux_y[0];
		    sBuffer_z[tile_pos.w] += sBuffer_redux_z[0];
		    tile_pos.w++;

		    sBuffer_x[tile_pos.w] += sBuffer_redux_x2[0];
		    sBuffer_y[tile_pos.w] += sBuffer_redux_y2[0];
		    sBuffer_z[tile_pos.w] += sBuffer_redux_z2[0];
		    tile_pos.w++;
		}
		__syncthreads();
		// ---------------------------------------------------------------------------------

	    }
	} // LOOP: 64 B-Spline Values for current voxel_cluster

    } // LOOP: voxel_clusters


    // ----------------------------------------------------------
    //                STAGE 3 IN POWERPOINT
    // ----------------------------------------------------------
    // By this point every voxel cluster within the tile has been
    // processed for every possible tile position (there are 64).
    // ----------------------------------------------------------
    // HERE, EACH WARP OPERATES ON A SINGLE TILE'S SET OF 64!!
    // ----------------------------------------------------------
    tileOffset = 64*blockIdxInGrid;

    tile_pos.x = 63 - threadIdx.x;

    int knot_num;

    knot_num = LUT_Knot[tileOffset + threadIdx.x];

    cond_x[ (64*knot_num) + tile_pos.x ] = sBuffer_x[threadIdx.x];
    cond_y[ (64*knot_num) + tile_pos.x ] = sBuffer_y[threadIdx.x];
    cond_z[ (64*knot_num) + tile_pos.x ] = sBuffer_z[threadIdx.x];
    // ----------------------------------------------------------

    // Done with tile.

    // END OF KERNEL
}



/**
 * This kernel partially computes the gradient by generating condensed dc_dv values.
 *
 * @warning It is required that input data tiles be aligned to 64 byte boundaries.
 *
 * @see CUDA_pad_64()
 * @see kernel_pad_64()
 *
 * @param cond_x Pointer to condensed dc_dv x-values
 * @param cond_y Pointer to condensed dc_dv y-values
 * @param cond_z Pointer to condensed dc_dv z-values
 * @param dc_dv_x Pointer to dc_dv x-values
 * @param dc_dv_y Pointer to dc_dv y-values
 * @param dc_dv_z Pointer to dc_dv z-values
 * @param LUT_Tile_Offsets Pointer to offset lookup table
 * @param LUT_Knot Pointer to linear knot indices
 * @param pad Amount of tile padding, in bytes
 * @param tile_dim Dimensions of input volume tiles
 * @param one_over_six The value 1/6
 *
 * @author: James A. Shackleford
 */
__global__ void kernel_bspline_mse_2_condense_64(
    float* cond_x,		// Return: condensed dc_dv_x values
    float* cond_y,		// Return: condensed dc_dv_y values
    float* cond_z,		// Return: condensed dc_dv_z values
    float* dc_dv_x,		// Input : dc_dv_x values
    float* dc_dv_y,		// Input : dc_dv_y values
    float* dc_dv_z,		// Input : dc_dv_z values
    int* LUT_Tile_Offsets,	// Input : tile offsets
    int* LUT_Knot,		// Input : linear knot indices
    int pad,		// Input : amount of tile padding
    int4 tile_dim,		// Input : dims of tiles
    float one_over_six)	// Input : Precomputed since GPU division is slow
{
    // NOTES
    // * Each threadblock contains 2 warps.
    // * Each set of 2 warps operates on only one tile
    // * Each tile is reduced to 64x3 single precision floating point values
    // * Each of the 64 values consists of 3 floats [x,y,z]
    // * Each of the 64 values relates to a different control knot
    // * Each set of 3 floats (there are 64 sets) are placed into a stream
    // * The stream is indexed into by an offset + [0,64].
    // * The offset is the knot number that the set of 3 floats influences
    // * Each warp will write to 64 different offsets

    int tileOffset;
    int voxel_cluster;
    int voxel_idx;
    float3 voxel_val;
    int3 voxel_loc;
    int4 tile_pos;
    float A,B,C,D;


    // -- Setup Thread Attributes -----------------------------
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    int myWarpId_inPair = threadIdxInGrid - 64*blockIdxInGrid;		// From 0 to 63
    // --------------------------------------------------------


    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: 3*threadsPerBlock*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer_x = (float*)sdata;			// sBuffer_x[64]
    float* sBuffer_y = (float*)&sBuffer_x[64];		// sBuffer_y[64]
    float* sBuffer_z = (float*)&sBuffer_y[64];		// sBuffer_z[64]
    float* sBuffer_redux_x = (float*)&sBuffer_z[64];	// sBuffer_redux_x[64]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[64];	// sBuffer_redux_y[64]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[64];	// sBuffer_redux_z[64]
    // --------------------------------------------------------


    // Clear Shared Memory!!
    sBuffer_x[myWarpId_inPair] = 0;
    sBuffer_y[myWarpId_inPair] = 0;
    sBuffer_z[myWarpId_inPair] = 0;


    // First, get the offset of where our tile starts in memory.
    tileOffset = LUT_Tile_Offsets[blockIdxInGrid];

    // Main Loop for Warp Work
    // (Here we condense a tile into 64x3 floats)
    for (voxel_cluster=0; voxel_cluster < tile_dim.w; voxel_cluster+=64)
    {

	// ----------------------------------------------------------
	//                  STAGE 1 IN POWERPOINT
	// ----------------------------------------------------------
	// Second, we pulldown the current voxel cluster.
	// Each thread in the warp pulls down 1 voxel (3 values)
	// ----------------------------------------------------------
	voxel_val.x = dc_dv_x[tileOffset + voxel_cluster + myWarpId_inPair];
	voxel_val.y = dc_dv_y[tileOffset + voxel_cluster + myWarpId_inPair];
	voxel_val.z = dc_dv_z[tileOffset + voxel_cluster + myWarpId_inPair];
	// ----------------------------------------------------------

	// Third, find the [x,y,z] location within the current tile
	// for the voxel this thread is processing.
	voxel_idx = (voxel_cluster + myWarpId_inPair);
	voxel_loc.x = voxel_idx % tile_dim.x;
	voxel_loc.y = ((voxel_idx - voxel_loc.x) / tile_dim.x) % tile_dim.y;
	voxel_loc.z = (((voxel_idx - voxel_loc.x) / tile_dim.x) / tile_dim.y) % tile_dim.z;

	// Fourth, we will perform all 64x3 calculations on the current voxel cluster.
	// (Every thead in the warp will be doing this at the same time for its voxel)

	tile_pos.w = 0;	// Current tile position within [0,63]

	for (tile_pos.z = 0; tile_pos.z < 4; tile_pos.z++)
	    for (tile_pos.y = 0; tile_pos.y < 4; tile_pos.y++)
		for (tile_pos.x = 0; tile_pos.x < 4; tile_pos.x++)
		{

		    // ---------------------------------------------------------------------------------
		    //                           STAGE 2 IN POWERPOINT
		    // ---------------------------------------------------------------------------------

		    // Clear Shared Memory!!
		    sBuffer_redux_x[myWarpId_inPair] = 0;
		    sBuffer_redux_y[myWarpId_inPair] = 0;
		    sBuffer_redux_z[myWarpId_inPair] = 0;

		    // Calculate the b-spline multiplier for this voxel @ this tile
		    // position relative to a given control knot.
		    A = obtain_spline_basis_function(one_over_six, tile_pos.x, voxel_loc.x, tile_dim.x);
		    B = obtain_spline_basis_function(one_over_six, tile_pos.y, voxel_loc.y, tile_dim.y);
		    C = obtain_spline_basis_function(one_over_six, tile_pos.z, voxel_loc.z, tile_dim.z);
		    D = A*B*C;

		    // Perform the multiplication and store to redux shared memory
		    sBuffer_redux_x[myWarpId_inPair] = voxel_val.x * D;
		    sBuffer_redux_y[myWarpId_inPair] = voxel_val.y * D;
		    sBuffer_redux_z[myWarpId_inPair] = voxel_val.z * D;
		    __syncthreads();

		    // All 64 dc_dv values in the current cluster have been processed
		    // for the current tile position (out of 64 total tile positions).
					
		    // We now perform a sum reduction on these 64 dc_dv values to
		    // condense the data down to one value.
		    for(unsigned int s = 32; s > 0; s >>= 1)
		    {
			if (myWarpId_inPair < s)
			{
			    sBuffer_redux_x[myWarpId_inPair] += sBuffer_redux_x[myWarpId_inPair + s];
			    sBuffer_redux_y[myWarpId_inPair] += sBuffer_redux_y[myWarpId_inPair + s];
			    sBuffer_redux_z[myWarpId_inPair] += sBuffer_redux_z[myWarpId_inPair + s];
			}

			// Wait for all threads in to complete the current tier.
			__syncthreads();
		    }

		    // We then accumulate this single condensed value into the element of
		    // shared memory that correlates to the current tile position.
		    if (myWarpId_inPair == 0)
		    {
			sBuffer_x[tile_pos.w] += sBuffer_redux_x[0];
			sBuffer_y[tile_pos.w] += sBuffer_redux_y[0];
			sBuffer_z[tile_pos.w] += sBuffer_redux_z[0];
		    }
		    __syncthreads();

		    // Continue to work on the current voxel cluster, but shift
		    // to the next tile position.
		    tile_pos.w++;
		    // ---------------------------------------------------------------------------------

		} // LOOP: 64 B-Spline Values for current voxel_cluster

    } // LOOP: voxel_clusters


    // ----------------------------------------------------------
    //                STAGE 3 IN POWERPOINT
    // ----------------------------------------------------------
    // By this point every voxel cluster within the tile has been
    // processed for every possible tile position (there are 64).
    //
    // Now it is time to put these 64 condensed values in their
    // proper places.  We will work off of myGlobalWarpNumber,
    // which is equal to the tile index, and myWarpId, which is
    // equal to the knot number [0,63].
    // ----------------------------------------------------------
    // HERE, EACH WARP OPERATES ON A SINGLE TILE'S SET OF 64!!
    // ----------------------------------------------------------
    tileOffset = 64*blockIdxInGrid;

    tile_pos.x = 63 - myWarpId_inPair;

    int knot_num;

    knot_num = LUT_Knot[tileOffset + myWarpId_inPair];

    cond_x[ (64*knot_num) + tile_pos.x ] = sBuffer_x[myWarpId_inPair];
    cond_y[ (64*knot_num) + tile_pos.x ] = sBuffer_y[myWarpId_inPair];
    cond_z[ (64*knot_num) + tile_pos.x ] = sBuffer_z[myWarpId_inPair];
    // ----------------------------------------------------------

    // Done with tile.

    // END OF KERNEL
}



/**
 * This kernel partially computes the gradient by generating condensed dc_dv values.
 *
 * @warning It is required that input data tiles be aligned to 32 byte boundaries.
 *
 * @see CUDA_pad_32()
 * @see kernel_pad_32()
 *
 * @param cond_x Pointer to condensed dc_dv x-values
 * @param cond_y Pointer to condensed dc_dv y-values
 * @param cond_z Pointer to condensed dc_dv z-values
 * @param dc_dv_x Pointer to dc_dv x-values
 * @param dc_dv_y Pointer to dc_dv y-values
 * @param dc_dv_z Pointer to dc_dv z-values
 * @param LUT_Tile_Offsets Pointer to offset lookup table
 * @param LUT_Knot Pointer to linear knot indices
 * @param pad Amount of tile padding, in bytes
 * @param tile_dim Dimensions of input volume tiles
 * @param one_over_six The value 1/6
 *
 * @author: James A. Shackleford
 */
__global__ void kernel_bspline_mse_2_condense(
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
    float one_over_six)	// Input : Precomputed since GPU division is slow
{
    // NOTES
    // * Each threadblock contains only 1 warp.
    // * Each warp operates on only one tile
    // * Each tile is reduced to 64x3 single precision floating point values
    // * Each of the 64 values consists of 3 floats [x,y,z]
    // * Each of the 64 values relates to a different control knot
    // * Each set of 3 floats (there are 64 sets) are placed into a stream
    // * The stream is indexed into by an offset + [0,64].
    // * The offset is the knot number that the set of 3 floats influences
    // * Each warp will write to 64 different offsets

    int tileOffset;
    int voxel_cluster;
    int voxel_idx;
    float3 voxel_val;
    int3 voxel_loc;
    int4 tile_pos;
    float A,B,C,D;


    // -- Setup Thread Attributes -----------------------------
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    int myGlobalWarpNumber = threadIdxInGrid / 32;	// Tile #
    int myWarpId = threadIdxInGrid - 32*myGlobalWarpNumber;	// 0 to 31
    // --------------------------------------------------------


    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: 3*threadsPerBlock*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer_x = (float*)sdata;			// sBuffer_x[64]
    float* sBuffer_y = (float*)&sBuffer_x[64];		// sBuffer_y[64]
    float* sBuffer_z = (float*)&sBuffer_y[64];		// sBuffer_z[64]
    float* sBuffer_redux_x = (float*)&sBuffer_z[64];	// sBuffer_redux_x[32]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[32];	// sBuffer_redux_y[32]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[32];	// sBuffer_redux_z[32]
    // --------------------------------------------------------


    // Clear Shared Memory!!
    sBuffer_x[myWarpId] = 0; sBuffer_x[myWarpId+32] = 0;
    sBuffer_y[myWarpId] = 0; sBuffer_y[myWarpId+32] = 0;
    sBuffer_z[myWarpId] = 0; sBuffer_z[myWarpId+32] = 0;


    // First, get the offset of where our tile starts in memory.
    //	tileOffset = tex1Dfetch(tex_LUT_Offsets, myGlobalWarpNumber);
    tileOffset = LUT_Tile_Offsets[myGlobalWarpNumber];

    // Main Loop for Warp Work
    // (Here we condense a tile into 64x3 floats)
    for (voxel_cluster=0; voxel_cluster < tile_dim.w; voxel_cluster+=32)
    {

	// ----------------------------------------------------------
	//                  STAGE 1 IN POWERPOINT
	// ----------------------------------------------------------
	// Second, we pulldown the current voxel cluster.
	// Each thread in the warp pulls down 1 voxel (3 values)
	// ----------------------------------------------------------
	voxel_val.x = dc_dv_x[tileOffset + voxel_cluster + myWarpId];
	voxel_val.y = dc_dv_y[tileOffset + voxel_cluster + myWarpId];
	voxel_val.z = dc_dv_z[tileOffset + voxel_cluster + myWarpId];
	// ----------------------------------------------------------

	// Third, find the [x,y,z] location within the current tile
	// for the voxel this thread is processing.
	voxel_idx = (voxel_cluster + myWarpId);
	voxel_loc.x = voxel_idx % tile_dim.x;
	voxel_loc.y = ((voxel_idx - voxel_loc.x) / tile_dim.x) % tile_dim.y;
	voxel_loc.z = (((voxel_idx - voxel_loc.x) / tile_dim.x) / tile_dim.y) % tile_dim.z;

	// Third, we will perform all 64x3 calculations on the current voxel cluster.
	// (Every thead in the warp will be doing this at the same time for its voxel)

	tile_pos.w = 0;	// Current tile position within [0,63]

	for (tile_pos.z = 0; tile_pos.z < 4; tile_pos.z++)
	    for (tile_pos.y = 0; tile_pos.y < 4; tile_pos.y++)
		for (tile_pos.x = 0; tile_pos.x < 4; tile_pos.x++)
		{

		    // ---------------------------------------------------------------------------------
		    //                           STAGE 2 IN POWERPOINT
		    // ---------------------------------------------------------------------------------

		    // Clear Shared Memory!!
		    sBuffer_redux_x[myWarpId] = 0;
		    sBuffer_redux_y[myWarpId] = 0;
		    sBuffer_redux_z[myWarpId] = 0;

		    // Calculate the b-spline multiplier for this voxel @ this tile
		    // position relative to a given control knot.
		    A = obtain_spline_basis_function(one_over_six, tile_pos.x, voxel_loc.x, tile_dim.x);
		    B = obtain_spline_basis_function(one_over_six, tile_pos.y, voxel_loc.y, tile_dim.y);
		    C = obtain_spline_basis_function(one_over_six, tile_pos.z, voxel_loc.z, tile_dim.z);
		    D = A*B*C;
					
		    // Perform the multiplication and store to redux shared memory
		    sBuffer_redux_x[myWarpId] = voxel_val.x * D;
		    sBuffer_redux_y[myWarpId] = voxel_val.y * D;
		    sBuffer_redux_z[myWarpId] = voxel_val.z * D;
		    __syncthreads();

		    // All 32 voxels in the current cluster have been processed
		    // for the current tile position (out of 64 total tile positions).
					
		    // We now perform a sum reduction on these 32 voxels to condense the
		    // data down to one value.
		    for(unsigned int s = 16; s > 0; s >>= 1)
		    {
			if (myWarpId < s)
			{
			    sBuffer_redux_x[myWarpId] += sBuffer_redux_x[myWarpId + s];
			    sBuffer_redux_y[myWarpId] += sBuffer_redux_y[myWarpId + s];
			    sBuffer_redux_z[myWarpId] += sBuffer_redux_z[myWarpId + s];
			}

			// Wait for all threads in to complete the current tier.
			__syncthreads();
		    }

		    // We then accumulate this single condensed value into the element of
		    // shared memory that correlates to the current tile position.
		    if (myWarpId == 0)
		    {
			sBuffer_x[tile_pos.w] += sBuffer_redux_x[0];
			sBuffer_y[tile_pos.w] += sBuffer_redux_y[0];
			sBuffer_z[tile_pos.w] += sBuffer_redux_z[0];
		    }
		    __syncthreads();

		    // Continue to work on the current voxel cluster, but shift
		    // to the next tile position.
		    tile_pos.w++;
		    // ---------------------------------------------------------------------------------

		} // LOOP: 64 B-Spline Values for current voxel_cluster

    } // LOOP: voxel_clusters


    // ----------------------------------------------------------
    //                STAGE 3 IN POWERPOINT
    // ----------------------------------------------------------
    // By this point every voxel cluster within the tile has been
    // processed for every possible tile position (there are 64).
    //
    // Now it is time to put these 64 condensed values in their
    // proper places.  We will work off of myGlobalWarpNumber,
    // which is equal to the tile index, and myWarpId, which is
    // equal to the knot number [0,63].
    // ----------------------------------------------------------
    // HERE, EACH WARP OPERATES ON A SINGLE TILE'S SET OF 64!!
    // ----------------------------------------------------------
    tileOffset = 64*myGlobalWarpNumber;

    tile_pos.x = 63 - myWarpId;
    tile_pos.y = 63 - (myWarpId + 32);

    int knot_num;

    knot_num = LUT_Knot[tileOffset + myWarpId];

    cond_x[ (64*knot_num) + tile_pos.x ] = sBuffer_x[myWarpId];
    cond_y[ (64*knot_num) + tile_pos.x ] = sBuffer_y[myWarpId];
    cond_z[ (64*knot_num) + tile_pos.x ] = sBuffer_z[myWarpId];

    knot_num = LUT_Knot[tileOffset + myWarpId + 32];

    cond_x[ (64*knot_num) + tile_pos.y ] = sBuffer_x[myWarpId + 32];
    cond_y[ (64*knot_num) + tile_pos.y ] = sBuffer_y[myWarpId + 32];
    cond_z[ (64*knot_num) + tile_pos.y ] = sBuffer_z[myWarpId + 32];
    // ----------------------------------------------------------

    // Done with tile.

    // END OF KERNEL


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // NOTE
    // LUT_knot[64*numTiles] contains the linear knot indicies and is organized as follows:
    //
    //    Tile 0                               Tile 1                               Tile N
    // +-----------+-----------+------------+-----------+-----------+------------+-----------+-----------+------------+
    // | knot_idx0 |    ...    | knot_idx63 | knot_idx0 |    ...    | knot_idx63 | knot_idx0 |    ...    | knot_idx63 |
    // +-----------+-----------+--=---------+-----------+-----------+------------+-----------+-----------+-----=------+
    //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// KERNEL: kernel_bspline_mse_2_reduce()
//
// * Each threadblock contains only 2 warps.
// * Each threadblock operates on 32 knots (1 at a time)
//
// * Each knot in a condense stream contains 64 single precision floats
// * Each knot is spread across the 3 condense streams [x,y,z]
// * The "high warp" will handle floats 32-63
// * The "low warp"  will handle floats  0-31
//
// * The 2 warps will work together to sum reduce the 64 floats to 1 float
// * The sum reduction result is stored in shared memory
//
// AUTHOR: James Shackleford
// DATE  : August 27th, 2009
////////////////////////////////////////////////////////////////////////////////
__global__ void
kernel_bspline_mse_2_reduce (
    float* grad,		// Return: interleaved dc_dp values
    float* cond_x,		// Input : condensed dc_dv_x values
    float* cond_y,		// Input : condensed dc_dv_y values
    float* cond_z)		// Input : condensed dc_dv_z values
{
    // -- Setup Thread Attributes -----------------------------
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    // --------------------------------------------------------

    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: ((3*64)+3)*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer = (float*)sdata;				// sBuffer[3]
    float* sBuffer_redux_x = (float*)&sBuffer[3];		// sBuffer_redux_x[64]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[64];	// sBuffer_redux_y[64]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[64];	// sBuffer_redux_z[64]
    // --------------------------------------------------------

    // Pull down the 64 condensed dc_dv values for the knot this warp pair is working on
    sBuffer_redux_x[threadIdx.x] = cond_x[64*blockIdxInGrid + threadIdx.x];
    sBuffer_redux_y[threadIdx.x] = cond_y[64*blockIdxInGrid + threadIdx.x];
    sBuffer_redux_z[threadIdx.x] = cond_z[64*blockIdxInGrid + threadIdx.x];

    // This thread barrier is very important!
    __syncthreads();
	
    // Perform sum reduction on the 64 condensed dc_dv values
    for(unsigned int s = 32; s > 0; s >>= 1)
    {
	if (threadIdx.x < s)
	{
	    sBuffer_redux_x[threadIdx.x] += sBuffer_redux_x[threadIdx.x + s];
	    sBuffer_redux_y[threadIdx.x] += sBuffer_redux_y[threadIdx.x + s];
	    sBuffer_redux_z[threadIdx.x] += sBuffer_redux_z[threadIdx.x + s];
	}

	// Wait for all threads in to complete the current tier.
	__syncthreads();
    }



    // Store 3 resulting floats into the output buffer (shared memory)
    // These 3 floats are the dc_dp value [x,y,z] for the current knot
    // This shared memory store is interleaved so that the final global
    // memory store will be coalaced.
    if (threadIdx.x == 0)
	sBuffer[0] = sBuffer_redux_x[0];
	
    if (threadIdx.x == 1)
	sBuffer[1] = sBuffer_redux_y[0];

    if (threadIdx.x == 2)
	sBuffer[2] = sBuffer_redux_z[0];

    // Prevent read before write race condition
    __syncthreads();


    if (threadIdx.x < 3)
	grad[3*blockIdxInGrid + threadIdx.x] = sBuffer[threadIdx.x];

    // END OF KERNEL 
}


//////////////////////////////////////////////////////////////////////////////
// KERNEL: bspline_cuda_score_j_mse_kernel1()
//
// This is idential to bspline_cuda_score_g_mse_kernel1() except it generates
// three seperate dc_dv streams: dc_dv_x, dc_dv_y, and dc_dv_z.
//
// This removes the need for deinterleaving the dc_dv stream before running
// the CUDA condense_64() kernel, which removes CUDA_deinterleave() from the
// execution chain.
//////////////////////////////////////////////////////////////////////////////
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
    int3   cdims,          // # of control points in (x,y,z)
    int    pad,
    float  *skipped        // # of voxels that fell outside the ROI
)
{
    extern __shared__ float sdata[]; 
	
    int3   fix_ijk;           // Index of the voxel in the fixed image (vox)
    int3   p;		      // Index of the tile within the volume (vox)
    int3   q;		      // Offset within the tile (measured in voxels)
    int    fv;		      // Index of voxel in linear image array
    int    pidx;	      // Index into c_lut
    int    qidx;	      // Index into q_lut
    int    cidx;	      // Index into the coefficient table

    float  P;
    float3 N;		      // Multiplier values
    float3 d;		      // B-spline deformation vector
    float  diff;

    float3 fix_xyz;           // Physical position of fixed image voxel (mm)
    float3 mov_xyz;           // Physical position of corresponding vox (mm)
    float3 mov_ijk;           // Index of corresponding vox (vox)
    int3 mov_ijk_floor;
    float3 mov_ijk_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int mvf;
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element_x;
    float* dc_dv_element_y;
    float* dc_dv_element_z;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, 
    // in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) 
	+ (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, 
    // based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) 
	+ threadIdxInBlock;

    // Allocate memory for the spline coefficients evaluated at 
    // indices 0, 1, 2, and 3 in the X, Y, and Z directions
    float *A = &sdata[12*threadIdxInBlock + 0];
    float *B = &sdata[12*threadIdxInBlock + 4];
    float *C = &sdata[12*threadIdxInBlock + 8];
    float ii, jj, kk;
    float t1, t2, t3; 
    float one_over_six = 1.0f/6.0f;

    // If the voxel lies outside the volume, do nothing.
    if (threadIdxInGrid < (fix_dim.x * fix_dim.y * fix_dim.z))
    {
	// Calculate the x, y, and z coordinate of the voxel within the volume.
	fix_ijk.z = threadIdxInGrid / (fix_dim.x * fix_dim.y);
	fix_ijk.y = (threadIdxInGrid 
	    - (fix_ijk.z * fix_dim.x * fix_dim.y)) / fix_dim.x;
	fix_ijk.x = threadIdxInGrid 
	    - fix_ijk.z * fix_dim.x * fix_dim.y 
	    - (fix_ijk.y * fix_dim.x);
			
	// Calculate the x, y, and z offsets of the tile that 
	// contains this voxel.
	p.x = fix_ijk.x / vox_per_rgn.x;
	p.y = fix_ijk.y / vox_per_rgn.y;
	p.z = fix_ijk.z / vox_per_rgn.z;

	// Calculate the x, y, and z offsets of the voxel within the tile.
	q.x = fix_ijk.x - p.x * vox_per_rgn.x;
	q.y = fix_ijk.y - p.y * vox_per_rgn.y;
	q.z = fix_ijk.z - p.z * vox_per_rgn.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if (fix_ijk.x < (roi_offset.x + roi_dim.x) || 
	    fix_ijk.y < (roi_offset.y + roi_dim.y) ||
	    fix_ijk.z < (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (fix_ijk.z * fix_dim.x * fix_dim.y) 
		+ (fix_ijk.y * fix_dim.x) + fix_ijk.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // pidx is the tile index for the tile of the current voxel
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element_x = &dc_dv_x[((vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z) + pad) * pidx];
	    dc_dv_element_y = &dc_dv_y[((vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z) + pad) * pidx];
	    dc_dv_element_z = &dc_dv_z[((vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z) + pad) * pidx];

	    // qidx is the local index of the voxel within the tile
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element_x = &dc_dv_element_x[qidx];
	    dc_dv_element_y = &dc_dv_element_y[qidx];
	    dc_dv_element_z = &dc_dv_element_z[qidx];

	    // Compute the q_lut values that pertain to this offset.
	    ii = ((float)q.x) / vox_per_rgn.x;
	    t3 = ii*ii*ii;
	    t2 = ii*ii;
	    t1 = ii;
	    A[0] = one_over_six * (- 1.0f * t3 + 3.0f * t2 - 3.0f * t1 + 1.0f);
	    A[1] = one_over_six * (+ 3.0f * t3 - 6.0f * t2             + 4.0f);
	    A[2] = one_over_six * (- 3.0f * t3 + 3.0f * t2 + 3.0f * t1 + 1.0f);
	    A[3] = one_over_six * (+ 1.0f * t3);

	    jj = ((float)q.y) / vox_per_rgn.y;
	    t3 = jj*jj*jj;
	    t2 = jj*jj;
	    t1 = jj;
	    B[0] = one_over_six * (- 1.0f * t3 + 3.0f * t2 - 3.0f * t1 + 1.0f);
	    B[1] = one_over_six * (+ 3.0f * t3 - 6.0f * t2             + 4.0f);
	    B[2] = one_over_six * (- 3.0f * t3 + 3.0f * t2 + 3.0f * t1 + 1.0f);
	    B[3] = one_over_six * (+ 1.0f * t3);

	    kk = ((float)q.z) / vox_per_rgn.z;
	    t3 = kk*kk*kk;
	    t2 = kk*kk;
	    t1 = kk;
	    C[0] = one_over_six * (- 1.0f * t3 + 3.0f * t2 - 3.0f * t1 + 1.0f);
	    C[1] = one_over_six * (+ 3.0f * t3 - 6.0f * t2             + 4.0f);
	    C[2] = one_over_six * (- 3.0f * t3 + 3.0f * t2 + 3.0f * t1 + 1.0f);
	    C[3] = one_over_six * (+ 1.0f * t3);

	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    // Compute the B-spline interpolant for the voxel
	    int3 t;
	    for (t.z = 0; t.z < 4; t.z++) {
		for (t.y = 0; t.y < 4; t.y++) {
		    for (t.x = 0; t.x < 4; t.x++) {

			// Calculate the index into the coefficients array.
			cidx = 3 * ((p.z + t.z) * cdims.x * cdims.y 
			    + (p.y + t.y) * cdims.x + (p.x + t.x));

			// Fetch the values for P, Ni, Nj, and Nk.
			P   = A[t.x] * B[t.y] * C[t.z];
			N.x = TEX_REF (coeff, cidx + 0);
			N.y = TEX_REF (coeff, cidx + 1);
			N.z = TEX_REF (coeff, cidx + 2);
			
			// Update the output (v) values.
			d.x += P * N.x;
			d.y += P * N.y;
			d.z += P * N.z;
		    }
		}
	    }

	    //-----------------------------------------------------------------
	    // Find correspondence in the moving image.
	    //-----------------------------------------------------------------

	    // Calculate the position of the voxel (in mm)
	    fix_xyz.x = fix_origin.x + (fix_spacing.x * fix_ijk.x);
	    fix_xyz.y = fix_origin.y + (fix_spacing.y * fix_ijk.y);
	    fix_xyz.z = fix_origin.z + (fix_spacing.z * fix_ijk.z);
			
	    // Calculate the corresponding voxel in the moving image (in mm)
	    mov_xyz.x = fix_xyz.x + d.x;
	    mov_xyz.y = fix_xyz.y + d.y;
	    mov_xyz.z = fix_xyz.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    mov_ijk.x = (mov_xyz.x - mov_origin.x) / mov_spacing.x;
	    mov_ijk.y = (mov_xyz.y - mov_origin.y) / mov_spacing.y;
	    mov_ijk.z = (mov_xyz.z - mov_origin.z) / mov_spacing.z;

	    // Check if the displaced voxel lies outside the 
	    // region of interest.
	    if ((mov_ijk.x < -0.5) || (mov_ijk.x > (mov_dim.x - 0.5)) 
		|| (mov_ijk.y < -0.5) || (mov_ijk.y > (mov_dim.y - 0.5)) 
		|| (mov_ijk.z < -0.5) || (mov_ijk.z > (mov_dim.z - 0.5)))
	    {
		// Count voxel as outside the ROI
		skipped[threadIdxInGrid]++;

	    } else {

		//-----------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------

		// Clamp and interpolate along the X axis.
		mov_ijk_floor.x = (int) floorf (mov_ijk.x);
		mov_ijk_round.x = rintf (mov_ijk.x);
		fx2 = mov_ijk.x - mov_ijk_floor.x;
		if (mov_ijk_floor.x < 0) {
		    mov_ijk_floor.x = 0;
		    mov_ijk_round.x = 0;
		    fx2 = 0.0f;
		}
		else if (mov_ijk_floor.x >= (mov_dim.x - 1)) {
		    mov_ijk_floor.x = mov_dim.x - 2;
		    mov_ijk_round.x = mov_dim.x - 1;
		    fx2 = 1.0f;
		}
		fx1 = 1.0f - fx2;

		// Clamp and interpolate along the Y axis.
		mov_ijk_floor.y = (int) floorf (mov_ijk.y);
		mov_ijk_round.y = rintf (mov_ijk.y);
		fy2 = mov_ijk.y - mov_ijk_floor.y;
		if (mov_ijk_floor.y < 0) {
		    mov_ijk_floor.y = 0;
		    mov_ijk_round.y = 0;
		    fy2 = 0.0f;
		}
		else if (mov_ijk_floor.y >= (mov_dim.y - 1)) {
		    mov_ijk_floor.y = mov_dim.y - 2;
		    mov_ijk_round.y = mov_dim.y - 1;
		    fy2 = 1.0f;
		}
		fy1 = 1.0f - fy2;

		// Clamp and intepolate along the Z axis.
		mov_ijk_floor.z = (int) floorf (mov_ijk.z);
		mov_ijk_round.z = rintf (mov_ijk.z);
		fz2 = mov_ijk.z - mov_ijk_floor.z;
		if (mov_ijk_floor.z < 0) {
		    mov_ijk_floor.z = 0;
		    mov_ijk_round.z = 0;
		    fz2 = 0.0f;
		}
		else if (mov_ijk_floor.z >= (mov_dim.z - 1)) {
		    mov_ijk_floor.z = mov_dim.z - 2;
		    mov_ijk_round.z = mov_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0f - fz2;

		//-----------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------
		mvf = (mov_ijk_floor.z * mov_dim.y + mov_ijk_floor.y) 
		    * mov_dim.x + mov_ijk_floor.x;

		m_x1y1z1 = fx1 * fy1 * fz1 * TEX_REF (moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * TEX_REF (moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * TEX_REF (moving_image, mvf + mov_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * TEX_REF (moving_image, mvf + mov_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * TEX_REF (moving_image, mvf + mov_dim.y * mov_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * TEX_REF (moving_image, mvf + mov_dim.y * mov_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * TEX_REF (moving_image, mvf + mov_dim.y * mov_dim.x + mov_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * TEX_REF (moving_image, mvf + mov_dim.y * mov_dim.x + mov_dim.x + 1);

		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		// Compute intensity difference.
#if PLM_DONT_INVERT_GRADIENT
		diff = m_val - TEX_REF (fixed_image, fv);
#else
		diff = TEX_REF (fixed_image, fv) - m_val;
#endif

		// Accumulate the score.
		score[threadIdxInGrid] = (diff * diff);

		//-----------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------
		// Compute spatial gradient using nearest neighbors.
		//		mvr = ((((int)mov_ijk_round.z * mov_dim.y) + (int)mov_ijk_round.y) * mov_dim.x) + (int)mov_ijk_round.x;
		// The above is commented out because mvr becomes too large
		// to be used as a GPU texture reference index.  See below
		// for the workaround using offsets

		// tex1Dfetch() uses 27-bits for indexing, which results in an
		// index overflow for large image volumes.  The following code
		// removes the usage of the 1D texture reference and attempts
		// to use several smaller indices and pointer arithmetic in
		// order to reduce the size of the index.
		float* big_fat_grad;

		big_fat_grad = &moving_grad[
		    3 * (int) mov_ijk_round.z * mov_dim.y * mov_dim.x];
		big_fat_grad = &big_fat_grad[
		    3 * (int) mov_ijk_round.y * mov_dim.x];
		big_fat_grad = &big_fat_grad[3 * (int) mov_ijk_round.x];

		dc_dv_element_x[0] = diff * big_fat_grad[0];
		dc_dv_element_y[0] = diff * big_fat_grad[1];
		dc_dv_element_z[0] = diff * big_fat_grad[2];

		// This code does not work for large image volumes > 512x512x170
		//		dc_dv_element_x[0] = diff * TEX_REF (moving_grad, 3 * (int)mvr + 0);
		//		dc_dv_element_y[0] = diff * TEX_REF (moving_grad, 3 * (int)mvr + 1);
		//		dc_dv_element_z[0] = diff * TEX_REF (moving_grad, 3 * (int)mvr + 2);
	    }
	}
    }
}




/******************************************************
* This function computes the spline basis function at 
* index 0, 1, 2, or 3 for a voxel 
Author: Naga Kandasamy
Date: 07 July 2009
*******************************************************/

__device__ float
obtain_spline_basis_function (float one_over_six,
    int t_idx, 
    int vox_idx, 
    int vox_per_rgn)
{
    float i = (float)vox_idx / vox_per_rgn;
    float C;
						
    switch(t_idx) {
    case 0:
	C = one_over_six * (- 1.0 * i*i*i + 3.0 * i*i - 3.0 * i + 1.0);
	break;
    case 1:
	C = one_over_six * (+ 3.0 * i*i*i - 6.0 * i*i            + 4.0);
	break;
    case 2:
	C = one_over_six * (- 3.0 * i*i*i + 3.0 * i*i + 3.0 * i + 1.0);
	break;
    case 3:
	C = one_over_six * (+ 1.0 * i*i*i);
	break;
    }
    return C;
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
	//		grad[threadIdxInGrid] = 2.0 * tex1Dfetch(tex_grad, threadIdxInGrid) / num_vox;
	grad[threadIdxInGrid] = 2.0 * grad[threadIdxInGrid] / num_vox;
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


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_initialize_j()
// 
// Initialize the GPU to execute bspline_cuda_score_j_mse().
//
// AUTHOR: James Shackleford
// DATE  : September 17, 2009
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_initialize_j(Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    BSPLINE_Xform* bxf,
    BSPLINE_Parms* parms)
{
    // Keep track of how much memory we allocated
    // in the GPU global memory.
    long unsigned GPU_Memory_Bytes = 0;

    // Tell the user we are busy copying information
    // to the device memory.
    printf ("Copying data to GPU global memory\n");

    // --- COPY FIXED IMAGE TO GPU GLOBAL -----------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->fixed_image_size = fixed->npix * fixed->pix_size;

    // Allocate memory in the GPU Global memory for the fixed
    // volume's voxel data. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->fixed_image. (fixed_image is a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->fixed_image_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc ((void**)&dev_ptrs->fixed_image, 
	dev_ptrs->fixed_image_size);
    checkCUDAError ("Failed to allocate memory for fixed image");
    printf(".");


    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy (dev_ptrs->fixed_image, fixed->img, 
	dev_ptrs->fixed_image_size, cudaMemcpyHostToDevice);
    checkCUDAError ("Failed to copy fixed image to GPU");
    printf(".");


    // Bind this to a texture reference
    cudaBindTexture (0, tex_fixed_image, dev_ptrs->fixed_image, 
	dev_ptrs->fixed_image_size);
    checkCUDAError ("Failed to bind dev_ptrs->fixed_image to texture reference!");
    printf(".");
	

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->fixed_image_size;
    // ----------------------------------------------------------


    // --- COPY MOVING IMAGE TO GPU GLOBAL ----------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->moving_image_size = moving->npix * moving->pix_size;

    // Allocate memory in the GPU Global memory for the moving
    // volume's voxel data. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->moving_image. (moving_image is a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->moving_image_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    checkCUDAError("Failed to allocate memory for moving image");
    printf(".");
	
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy( dev_ptrs->moving_image, moving->img, dev_ptrs->moving_image_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy moving image to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_image, dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    checkCUDAError("Failed to bind dev_ptrs->moving_image to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->moving_image_size;
    // ----------------------------------------------------------


    // --- COPY MOVING GRADIENT TO GPU GLOBAL -------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->moving_grad_size = moving_grad->npix * moving_grad->pix_size;

    // Allocate memory in the GPU Global memory for the moving grad
    // volume's data. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->moving_grad. (moving_grad is a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->moving_grad_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    checkCUDAError("Failed to allocate memory for moving grad");
    printf(".");
	
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    // (Note the pointer dereference)
    cudaMemcpy( dev_ptrs->moving_grad, moving_grad->img, dev_ptrs->moving_grad_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy moving grad to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_grad, dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    checkCUDAError("Failed to bind dev_ptrs->moving_image to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->moving_grad_size;
    // ----------------------------------------------------------


    // --- ALLOCATE COEFFICIENT LUT IN GPU GLOBAL ---------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->coeff_size = sizeof(float) * bxf->num_coeff;

    // Allocate memory in the GPU Global memory for the 
    // coefficient LUT. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->coeff. (coeff is a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->coeff_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->coeff, dev_ptrs->coeff_size);
    checkCUDAError("Failed to allocate memory for dev_ptrs->coeff");
    printf(".");


    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->coeff, 0, dev_ptrs->coeff_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_coeff, dev_ptrs->coeff, dev_ptrs->coeff_size);
    checkCUDAError("Failed to bind dev_ptrs->coeff to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->coeff_size;
    // ----------------------------------------------------------


    // --- ALLOCATE SCORE IN GPU GLOBAL -------------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->score_size = sizeof(float) * fixed->npix;
    dev_ptrs->skipped_size = sizeof(float) * fixed->npix;

    // Allocate memory in the GPU Global memory for the 
    // "Score". The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->score. (scoreis a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->score_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->score, dev_ptrs->score_size);
    printf(".");

    cudaMalloc((void**)&dev_ptrs->skipped, dev_ptrs->skipped_size);
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->score, 0, dev_ptrs->score_size);
    cudaMemset(dev_ptrs->skipped, 0, dev_ptrs->skipped_size);

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->score_size;
    GPU_Memory_Bytes += dev_ptrs->skipped_size;
    // ----------------------------------------------------------


    // --- ALLOCATE GRAD IN GPU GLOBAL --------------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->grad_size = sizeof(float) * bxf->num_coeff;

    // Allocate memory in the GPU Global memory for the 
    // grad. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->grad. (grad is a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->grad_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->grad, dev_ptrs->grad_size);
    printf(".");


    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_grad, dev_ptrs->grad, dev_ptrs->grad_size);
    checkCUDAError("Failed to bind dev_ptrs->grad to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->grad_size;
    // ----------------------------------------------------------


    // --- ALLOCATE GRAD_TEMP IN GPU GLOBAL ---------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->grad_temp_size = sizeof(float) * bxf->num_coeff;

    // Allocate memory in the GPU Global memory for the 
    // grad_temp. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->grad_temp. (grad_temp is a pointer)
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->grad_temp_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->grad_temp, dev_ptrs->grad_temp_size);
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->grad_temp, 0, dev_ptrs->grad_temp_size);

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->grad_temp_size;
    // ----------------------------------------------------------


    // --- ALLOCATE dc_dv_x,y,z IN GPU GLOBAL -------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    //int num_voxels = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];

    int3 vol_dim;
    vol_dim.x = fixed->dim[0];
    vol_dim.y = fixed->dim[1];
    vol_dim.z = fixed->dim[2];

    int3 tile_dim;
    tile_dim.x = bxf->vox_per_rgn[0];
    tile_dim.y = bxf->vox_per_rgn[1];
    tile_dim.z = bxf->vox_per_rgn[2];

    int4 num_tile;
    num_tile.x = (vol_dim.x+tile_dim.x-1) / tile_dim.x;
    num_tile.y = (vol_dim.y+tile_dim.y-1) / tile_dim.y;
    num_tile.z = (vol_dim.z+tile_dim.z-1) / tile_dim.z;
    num_tile.w = num_tile.x * num_tile.y * num_tile.z;

    int tile_padding = 64 - ((tile_dim.x * tile_dim.y * tile_dim.z) % 64);
    int tile_bytes = (tile_dim.x * tile_dim.y * tile_dim.z);

    dev_ptrs->dc_dv_x_size = ((tile_bytes + tile_padding) * num_tile.w) * sizeof(float);
    dev_ptrs->dc_dv_y_size = dev_ptrs->dc_dv_x_size;
    dev_ptrs->dc_dv_z_size = dev_ptrs->dc_dv_x_size;

    // Allocate memory in the GPU Global memory for the 
    // deinterleaved dc_dv arrays. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->dc_dv_X. 
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->dc_dv_x_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->dc_dv_x, dev_ptrs->dc_dv_x_size);
    GPU_Memory_Bytes += dev_ptrs->dc_dv_x_size;
    checkCUDAError("cudaMalloc(): dev_ptrs->dc_dv_x");
    printf(".");
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->dc_dv_y_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->dc_dv_y, dev_ptrs->dc_dv_y_size);
    GPU_Memory_Bytes += dev_ptrs->dc_dv_y_size;
    checkCUDAError("cudaMalloc(): dev_ptrs->dc_dv_y");
    printf(".");
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->dc_dv_z_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->dc_dv_z, dev_ptrs->dc_dv_z_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->dc_dv_z");
    GPU_Memory_Bytes += dev_ptrs->dc_dv_z_size;
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);

    // Increment the GPU memory byte counter
    // ----------------------------------------------------------


    // --- ALLOCATE TILE OFFSET LUT IN GPU GLOBAL ---------------
    int* offsets = calc_offsets(bxf->vox_per_rgn, bxf->cdims);

    //	int vox_per_tile = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    //	int pad = 64 - (vox_per_tile % 64);

    dev_ptrs->LUT_Offsets_size = num_tiles*sizeof(int);

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->LUT_Offsets_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->LUT_Offsets");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Offsets, offsets, dev_ptrs->LUT_Offsets_size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy(): offsets --> dev_ptrs->LUT_Offsets");
    cudaBindTexture(0, tex_LUT_Offsets, dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);

    free (offsets);

    GPU_Memory_Bytes += dev_ptrs->LUT_Offsets_size;
    // ----------------------------------------------------------

    // --- ALLOCATE KNOT LUT IN GPU GLOBAL ----------------------
    dev_ptrs->LUT_Knot_size = 64*num_tiles*sizeof(int);

    int* local_set_of_64 = (int*)malloc(64*sizeof(int));
    int* LUT_Knot = (int*)malloc(dev_ptrs->LUT_Knot_size);

    int i,j;
    for (i = 0; i < num_tiles; i++)
    {
	find_knots(local_set_of_64, i, bxf->rdims, bxf->cdims);
	for (j = 0; j < 64; j++)
	    LUT_Knot[64*i + j] = local_set_of_64[j];
    }
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->LUT_Knot_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->LUT_Knot");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Knot, LUT_Knot, dev_ptrs->LUT_Knot_size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy(): LUT_Knot --> dev_ptrs->LUT_Knot");

    //	cudaBindTexture(0, tex_LUT_Knot, dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    //	checkCUDAError("cudaBindTexture(): dev_ptrs->LUT_Knot");

    free (local_set_of_64);
    free (LUT_Knot);

    GPU_Memory_Bytes += dev_ptrs->LUT_Knot_size;
    // ----------------------------------------------------------

    // --- ALLOCATE CONDENSED dc_dv VECTORS IN GPU GLOBAL -------
    dev_ptrs->cond_x_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_y_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_z_size = 64*bxf->num_knots*sizeof(float);

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->cond_x_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->cond_x, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->cond_x");
    printf(".");

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->cond_y_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->cond_y, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->cond_y");
    printf(".");

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->cond_z_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->cond_z, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->cond_z");
    printf(".");

    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_z");

    GPU_Memory_Bytes += dev_ptrs->cond_x_size;
    GPU_Memory_Bytes += dev_ptrs->cond_y_size;
    GPU_Memory_Bytes += dev_ptrs->cond_z_size;
    // ----------------------------------------------------------

    // --- GENERATE B-SPLINE LOOK UP TABLE ----------------------
    dev_ptrs->LUT_Bspline_x_size = 4*bxf->vox_per_rgn[0]* sizeof(float);
    dev_ptrs->LUT_Bspline_y_size = 4*bxf->vox_per_rgn[1]* sizeof(float);
    dev_ptrs->LUT_Bspline_z_size = 4*bxf->vox_per_rgn[2]* sizeof(float);
    float* LUT_Bspline_x = (float*)malloc(dev_ptrs->LUT_Bspline_x_size);
    float* LUT_Bspline_y = (float*)malloc(dev_ptrs->LUT_Bspline_y_size);
    float* LUT_Bspline_z = (float*)malloc(dev_ptrs->LUT_Bspline_z_size);

    for (j = 0; j < 4; j++)
    {
	for (i = 0; i < bxf->vox_per_rgn[0]; i++)
	    LUT_Bspline_x[j*bxf->vox_per_rgn[0] + i] = CPU_obtain_spline_basis_function (j, i, bxf->vox_per_rgn[0]);

	for (i = 0; i < bxf->vox_per_rgn[1]; i++)
	    LUT_Bspline_y[j*bxf->vox_per_rgn[1] + i] = CPU_obtain_spline_basis_function (j, i, bxf->vox_per_rgn[1]);

	for (i = 0; i < bxf->vox_per_rgn[2]; i++)
	    LUT_Bspline_z[j*bxf->vox_per_rgn[2] + i] = CPU_obtain_spline_basis_function (j, i, bxf->vox_per_rgn[2]);
    }
	
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->LUT_Bspline_x_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->LUT_Bspline_x, dev_ptrs->LUT_Bspline_x_size);
    GPU_Memory_Bytes += dev_ptrs->LUT_Bspline_x_size;
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->LUT_Bspline_y_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->LUT_Bspline_y, dev_ptrs->LUT_Bspline_y_size);
    GPU_Memory_Bytes += dev_ptrs->LUT_Bspline_y_size;
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->LUT_Bspline_z_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->LUT_Bspline_z, dev_ptrs->LUT_Bspline_z_size);
    GPU_Memory_Bytes += dev_ptrs->LUT_Bspline_z_size;

    cudaMemcpy(dev_ptrs->LUT_Bspline_x, LUT_Bspline_x, dev_ptrs->LUT_Bspline_x_size, cudaMemcpyHostToDevice);
    printf(".");
    cudaMemcpy(dev_ptrs->LUT_Bspline_y, LUT_Bspline_y, dev_ptrs->LUT_Bspline_y_size, cudaMemcpyHostToDevice);
    printf(".");
    cudaMemcpy(dev_ptrs->LUT_Bspline_z, LUT_Bspline_z, dev_ptrs->LUT_Bspline_z_size, cudaMemcpyHostToDevice);
    printf(".");

    free (LUT_Bspline_x);
    free (LUT_Bspline_y);
    free (LUT_Bspline_z);

    cudaBindTexture(0, tex_LUT_Bspline_x, dev_ptrs->LUT_Bspline_x, dev_ptrs->LUT_Bspline_x_size);
    printf(".");
    cudaBindTexture(0, tex_LUT_Bspline_y, dev_ptrs->LUT_Bspline_y, dev_ptrs->LUT_Bspline_y_size);
    printf(".");
    cudaBindTexture(0, tex_LUT_Bspline_z, dev_ptrs->LUT_Bspline_z, dev_ptrs->LUT_Bspline_z_size);
    printf(".");

    // ----------------------------------------------------------

    // Inform user we are finished.
    printf("done.\n");

    // Report global memory allocation.
    printf("  Allocated: %ld MB\n", GPU_Memory_Bytes / 1048576);

}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_initialize_i()
// 
// Initialize the GPU to execute bspline_cuda_score_i_mse().
//
// AUTHOR: James Shackleford
// DATE  : September 16, 2009
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_initialize_i(Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    BSPLINE_Xform* bxf,
    BSPLINE_Parms* parms)
{
    // Keep track of how much memory we allocated
    // in the GPU global memory.
    int GPU_Memory_Bytes = 0;
    //	int temp;

    // Tell the user we are busy copying information
    // to the device memory.
    printf("Copying data to GPU global memory");

    // --- COPY FIXED IMAGE TO GPU GLOBAL -----------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->fixed_image_size = fixed->npix * fixed->pix_size;

    // Allocate memory in the GPU Global memory for the fixed
    // volume's voxel data. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->fixed_image. (fixed_image is a pointer)
    cudaMalloc((void**)&dev_ptrs->fixed_image, dev_ptrs->fixed_image_size);
    checkCUDAError("Failed to allocate memory for fixed image");
    printf(".");


    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy( dev_ptrs->fixed_image, fixed->img, dev_ptrs->fixed_image_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy fixed image to GPU");
    printf(".");


    // Bind this to a texture reference
    cudaBindTexture(0, tex_fixed_image, dev_ptrs->fixed_image, dev_ptrs->fixed_image_size);
    checkCUDAError("Failed to bind dev_ptrs->fixed_image to texture reference!");
    printf(".");
	

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->fixed_image_size;
    // ----------------------------------------------------------


    // --- COPY MOVING IMAGE TO GPU GLOBAL ----------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->moving_image_size = moving->npix * moving->pix_size;

    // Allocate memory in the GPU Global memory for the moving
    // volume's voxel data. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->moving_image. (moving_image is a pointer)
    cudaMalloc((void**)&dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    checkCUDAError("Failed to allocate memory for moving image");
    printf(".");
	
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy( dev_ptrs->moving_image, moving->img, dev_ptrs->moving_image_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy moving image to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_image, dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    checkCUDAError("Failed to bind dev_ptrs->moving_image to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->moving_image_size;
    // ----------------------------------------------------------


    // --- COPY MOVING GRADIENT TO GPU GLOBAL -------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->moving_grad_size = moving_grad->npix * moving_grad->pix_size;

    // Allocate memory in the GPU Global memory for the moving grad
    // volume's data. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->moving_grad. (moving_grad is a pointer)
    cudaMalloc((void**)&dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    checkCUDAError("Failed to allocate memory for moving grad");
    printf(".");
	
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    // (Note the pointer dereference)
    cudaMemcpy( dev_ptrs->moving_grad, moving_grad->img, dev_ptrs->moving_grad_size, cudaMemcpyHostToDevice);
    checkCUDAError("Failed to copy moving grad to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_grad, dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    checkCUDAError("Failed to bind dev_ptrs->moving_image to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->moving_grad_size;
    // ----------------------------------------------------------


    // --- ALLOCATE COEFFICIENT LUT IN GPU GLOBAL ---------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->coeff_size = sizeof(float) * bxf->num_coeff;

    // Allocate memory in the GPU Global memory for the 
    // coefficient LUT. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->coeff. (coeff is a pointer)
    cudaMalloc((void**)&dev_ptrs->coeff, dev_ptrs->coeff_size);
    checkCUDAError("Failed to allocate memory for dev_ptrs->coeff");
    printf(".");


    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->coeff, 0, dev_ptrs->coeff_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_coeff, dev_ptrs->coeff, dev_ptrs->coeff_size);
    checkCUDAError("Failed to bind dev_ptrs->coeff to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->coeff_size;
    // ----------------------------------------------------------


    // --- ALLOCATE SCORE IN GPU GLOBAL -------------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->score_size = sizeof(float) * fixed->npix;

    // Allocate memory in the GPU Global memory for the 
    // "Score". The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->score. (scoreis a pointer)
    cudaMalloc((void**)&dev_ptrs->score, dev_ptrs->score_size);
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->score, 0, dev_ptrs->score_size);

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->score_size;
    // ----------------------------------------------------------


    // --- ALLOCATE dc_dv IN GPU GLOBAL -------------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->dc_dv_size = 3 * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2]
	* bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float);

    // Allocate memory in the GPU Global memory for dc_dv
    // The pointer to this area of GPU global memory will
    // be returned and placed into dev_ptrs->dc_dv. (dc_dv is a pointer)
    cudaMalloc((void**)&dev_ptrs->dc_dv, dev_ptrs->dc_dv_size);
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->dc_dv, 0, dev_ptrs->dc_dv_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_dc_dv, dev_ptrs->dc_dv, dev_ptrs->dc_dv_size);
    checkCUDAError("Failed to bind dev_ptrs->dc_dv to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->dc_dv_size;
    // ----------------------------------------------------------


    // --- ALLOCATE GRAD IN GPU GLOBAL --------------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->grad_size = sizeof(float) * bxf->num_coeff;

    // Allocate memory in the GPU Global memory for the 
    // grad. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->grad. (grad is a pointer)
    cudaMalloc((void**)&dev_ptrs->grad, dev_ptrs->grad_size);
    printf(".");


    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_grad, dev_ptrs->grad, dev_ptrs->grad_size);
    checkCUDAError("Failed to bind dev_ptrs->grad to texture reference!");
    printf(".");

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->grad_size;
    // ----------------------------------------------------------


    // --- ALLOCATE GRAD_TEMP IN GPU GLOBAL ---------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->grad_temp_size = sizeof(float) * bxf->num_coeff;

    // Allocate memory in the GPU Global memory for the 
    // grad_temp. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->grad_temp. (grad_temp is a pointer)
    cudaMalloc((void**)&dev_ptrs->grad_temp, dev_ptrs->grad_temp_size);
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->grad_temp, 0, dev_ptrs->grad_temp_size);

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->grad_temp_size;
    // ----------------------------------------------------------


    // --- ALLOCATE dc_dv_x,y,z IN GPU GLOBAL -------------------
    // Calculate space requirements for the allocation
    // and tuck it away for later...
    dev_ptrs->dc_dv_x_size = dev_ptrs->dc_dv_size / 3;
    dev_ptrs->dc_dv_y_size = dev_ptrs->dc_dv_x_size;
    dev_ptrs->dc_dv_z_size = dev_ptrs->dc_dv_x_size;

    // Allocate memory in the GPU Global memory for the 
    // deinterleaved dc_dv arrays. The pointer to this area of GPU
    // global memory will be returned and placed into
    // dev_parms->dc_dv_X. 
    cudaMalloc((void**)&dev_ptrs->dc_dv_x, dev_ptrs->dc_dv_x_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->dc_dv_x");
    printf(".");
    cudaMalloc((void**)&dev_ptrs->dc_dv_y, dev_ptrs->dc_dv_y_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->dc_dv_y");
    printf(".");
    cudaMalloc((void**)&dev_ptrs->dc_dv_z, dev_ptrs->dc_dv_z_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->dc_dv_z");
    printf(".");

    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);

    // Increment the GPU memory byte counter
    GPU_Memory_Bytes += dev_ptrs->dc_dv_x_size;
    GPU_Memory_Bytes += dev_ptrs->dc_dv_y_size;
    GPU_Memory_Bytes += dev_ptrs->dc_dv_z_size;
    // ----------------------------------------------------------


    // --- ALLOCATE TILE OFFSET LUT IN GPU GLOBAL ---------------
    int* offsets = calc_offsets(bxf->vox_per_rgn, bxf->cdims);

    //	int vox_per_tile = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    //	int pad = 64 - (vox_per_tile % 64);

    dev_ptrs->LUT_Offsets_size = num_tiles*sizeof(int);

    cudaMalloc((void**)&dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->LUT_Offsets");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Offsets, offsets, dev_ptrs->LUT_Offsets_size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy(): offsets --> dev_ptrs->LUT_Offsets");
    cudaBindTexture(0, tex_LUT_Offsets, dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);

    free (offsets);

    GPU_Memory_Bytes += dev_ptrs->LUT_Offsets_size;
    // ----------------------------------------------------------

    // --- ALLOCATE KNOT LUT IN GPU GLOBAL ----------------------
    dev_ptrs->LUT_Knot_size = 64*num_tiles*sizeof(int);

    int* local_set_of_64 = (int*)malloc(64*sizeof(int));
    int* LUT_Knot = (int*)malloc(dev_ptrs->LUT_Knot_size);

    int i,j;
    for (i = 0; i < num_tiles; i++)
    {
	find_knots(local_set_of_64, i, bxf->rdims, bxf->cdims);
	for (j = 0; j < 64; j++)
	    LUT_Knot[64*i + j] = local_set_of_64[j];
    }
    cudaMalloc((void**)&dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->LUT_Knot");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Knot, LUT_Knot, dev_ptrs->LUT_Knot_size, cudaMemcpyHostToDevice);
    checkCUDAError("cudaMemcpy(): LUT_Knot --> dev_ptrs->LUT_Knot");

    //	cudaBindTexture(0, tex_LUT_Knot, dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    //	checkCUDAError("cudaBindTexture(): dev_ptrs->LUT_Knot");

    free (local_set_of_64);
    free (LUT_Knot);

    GPU_Memory_Bytes += dev_ptrs->LUT_Knot_size;
    // ----------------------------------------------------------

    // --- ALLOCATE CONDENSED dc_dv VECTORS IN GPU GLOBAL -------
    dev_ptrs->cond_x_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_y_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_z_size = 64*bxf->num_knots*sizeof(float);

    cudaMalloc((void**)&dev_ptrs->cond_x, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->cond_x");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->cond_y, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->cond_y");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->cond_z, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMalloc(): dev_ptrs->cond_z");
    printf(".");

    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->cond_z");

    GPU_Memory_Bytes += dev_ptrs->cond_x_size;
    GPU_Memory_Bytes += dev_ptrs->cond_y_size;
    GPU_Memory_Bytes += dev_ptrs->cond_z_size;
    // ----------------------------------------------------------

    // Inform user we are finished.
    printf("done.\n");

    // Report global memory allocation.
    printf("  Allocated: %d MB\n", GPU_Memory_Bytes / 1048576);

}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_clean_up_j()
//
// AUTHOR: James Shackleford
// DATE  : September 11th, 2009
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_clean_up_j(Dev_Pointers_Bspline* dev_ptrs)
{
    cudaUnbindTexture(tex_fixed_image);
    cudaUnbindTexture(tex_moving_image);
    cudaUnbindTexture(tex_moving_grad);
    cudaUnbindTexture(tex_coeff);
    cudaUnbindTexture(tex_grad);
    cudaUnbindTexture(tex_LUT_Offsets);
    cudaUnbindTexture(tex_LUT_Bspline_x);
    cudaUnbindTexture(tex_LUT_Bspline_y);
    cudaUnbindTexture(tex_LUT_Bspline_z);
	
    cudaFree(dev_ptrs->fixed_image);
    cudaFree(dev_ptrs->moving_image);
    cudaFree(dev_ptrs->moving_grad);
    cudaFree(dev_ptrs->coeff);
    cudaFree(dev_ptrs->score);
    cudaFree(dev_ptrs->grad);
    cudaFree(dev_ptrs->grad_temp);
    cudaFree(dev_ptrs->dc_dv_x);
    cudaFree(dev_ptrs->dc_dv_y);
    cudaFree(dev_ptrs->dc_dv_z);
    cudaFree(dev_ptrs->LUT_Offsets);
    cudaFree(dev_ptrs->LUT_Knot);
    cudaFree(dev_ptrs->cond_x);
    cudaFree(dev_ptrs->cond_y);
    cudaFree(dev_ptrs->cond_z);
    cudaFree(dev_ptrs->LUT_Bspline_x);
    cudaFree(dev_ptrs->LUT_Bspline_y);
    cudaFree(dev_ptrs->LUT_Bspline_z);
    cudaFree(dev_ptrs->skipped);
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_clean_up_i()
//
// AUTHOR: James Shackleford
// DATE  : September 11th, 2009
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_clean_up_i(Dev_Pointers_Bspline* dev_ptrs)
{

    cudaUnbindTexture(tex_LUT_Offsets);

    cudaFree(dev_ptrs->fixed_image);
    cudaFree(dev_ptrs->moving_image);
    cudaFree(dev_ptrs->moving_grad);
    cudaFree(dev_ptrs->coeff);
    cudaFree(dev_ptrs->score);
    cudaFree(dev_ptrs->dc_dv);
    cudaFree(dev_ptrs->dc_dv_x);
    cudaFree(dev_ptrs->dc_dv_y);
    cudaFree(dev_ptrs->dc_dv_z);
    cudaFree(dev_ptrs->cond_x);
    cudaFree(dev_ptrs->cond_y);
    cudaFree(dev_ptrs->cond_z);
    cudaFree(dev_ptrs->grad);
    cudaFree(dev_ptrs->grad_temp);
    cudaFree(dev_ptrs->LUT_Knot);
    cudaFree(dev_ptrs->LUT_Offsets);
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_clean_up_h()
//
// AUTHOR: James Shackleford
// DATE  : September 11th, 2009
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_clean_up_h(Dev_Pointers_Bspline* dev_ptrs)
{
    cudaFree(dev_ptrs->fixed_image);
    cudaFree(dev_ptrs->moving_image);
    cudaFree(dev_ptrs->moving_grad);
    cudaFree(dev_ptrs->coeff);
    cudaFree(dev_ptrs->score);
    cudaFree(dev_ptrs->dc_dv);
    cudaFree(dev_ptrs->dc_dv_x);
    cudaFree(dev_ptrs->dc_dv_y);
    cudaFree(dev_ptrs->dc_dv_z);
    cudaFree(dev_ptrs->cond_x);
    cudaFree(dev_ptrs->cond_y);
    cudaFree(dev_ptrs->cond_z);
    cudaFree(dev_ptrs->grad);
    cudaFree(dev_ptrs->grad_temp);
    cudaFree(dev_ptrs->LUT_Knot);
    cudaFree(dev_ptrs->LUT_Offsets);
}

////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_h_push_coeff_lut()
//
// This function overwries the coefficient LUT to the GPU global
// memory with the new coefficient LUT in preparation for
// the next iteration of score calculation.
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_h_push_coeff_lut(Dev_Pointers_Bspline* dev_ptrs, BSPLINE_Xform* bxf)
{
    // Copy the coefficient LUT to the GPU.
    cudaMemcpy(dev_ptrs->coeff, bxf->coeff, dev_ptrs->coeff_size, cudaMemcpyHostToDevice);
    checkCUDAError("[Kernel Panic!] Failed to copy coefficient LUT to GPU");
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_h_clear_score()
//
// This function sets all elements in the score (located on the GPU) to zero
// in preparation for the next iteration of the kernel.
////////////////////////////////////////////////////////////////////////////////
extern "C" void bspline_cuda_h_clear_score(Dev_Pointers_Bspline* dev_ptrs) 
{
    cudaMemset(dev_ptrs->score, 0, dev_ptrs->score_size);
    checkCUDAError("Failed to clear the score stream on GPU\n");
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_h_clear_grad()
//
// This function sets all elemtns in the gradients (located on the GPU) to
// zero in preparation for the next iteration of the kernel.
////////////////////////////////////////////////////////////////////////////////
extern "C" void bspline_cuda_h_clear_grad(Dev_Pointers_Bspline* dev_ptrs) 
{
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    checkCUDAError("Failed to clear the grad stream on GPU\n");
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_deinterleave()
//
// KERNELS INVOKED:
//   kernel_deinterleave()
//
// AUTHOR: James Shackleford
//   DATE: 22 July, 2009
////////////////////////////////////////////////////////////////////////////////
void CUDA_deinterleave(
    int num_values,
    float* input,
    float* out_x,
    float* out_y,
    float* out_z)
{

    // --- INITIALIZE GRID --------------------------------------
    int i;
    int warps_per_block = 3;	// This cannot be changed.
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = (num_values + threads_per_block - 1) / threads_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);
	
    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_deinterleave() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_deinterleave() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    int smemSize = 2*threads_per_block*sizeof(float);
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    kernel_deinterleave<<<dimGrid, dimBlock, smemSize>>>(
	num_values,
	input,
	out_x,
	out_y,
	out_z
    );
    // ----------------------------------------------------------

    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_deinterleave()");
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_pad_64()
//
// KERNELS INVOKED:
//   kernel_pad_64()
//
// AUTHOR: James Shackleford
//   DATE: 16 September, 2009
////////////////////////////////////////////////////////////////////////////////
extern "C" void CUDA_pad_64(
    float** input,
    int* vol_dims,
    int* tile_dims)
{

    // --- CALCULATE THINGS NEEDED BY THIS STUB -----------------
    int3 vol_dim;
    vol_dim.x = vol_dims[0];
    vol_dim.y = vol_dims[1];
    vol_dim.z = vol_dims[2];

    int3 tile_dim;
    tile_dim.x = tile_dims[0];
    tile_dim.y = tile_dims[1];
    tile_dim.z = tile_dims[2];

    int num_voxels = vol_dim.x * vol_dim.y * vol_dim.z;

    int4 num_tiles;
    num_tiles.x = (vol_dim.x+tile_dim.x-1) / tile_dim.x;
    num_tiles.y = (vol_dim.y+tile_dim.y-1) / tile_dim.y;
    num_tiles.z = (vol_dim.z+tile_dim.z-1) / tile_dim.z;
    num_tiles.w = num_tiles.x * num_tiles.y * num_tiles.z;

    int tile_padding = 64 - ((tile_dim.x * tile_dim.y * tile_dim.z) % 64);
    int tile_bytes = (tile_dim.x * tile_dim.y * tile_dim.z);

    int output_size = (tile_bytes + tile_padding) * num_tiles.w;
    // ----------------------------------------------------------



    // --- ALLOCATE GPU GLOBAL MEMORY FOR OUTPUT ----------------
    float* tmp_output;
    cudaMalloc((void**)&tmp_output, output_size*sizeof(float));
    cudaMemset(tmp_output, 0, output_size*sizeof(float));
    // ----------------------------------------------------------

    // --- INITIALIZE GRID --------------------------------------
    int i;
    int warps_per_block = 4;
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = (num_voxels+threads_per_block-1) / threads_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_pad_64() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    kernel_pad_64<<<dimGrid, dimBlock>>>(
	*input,
	tmp_output,
	vol_dim,
	tile_dim
    );
    // ----------------------------------------------------------

    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_pad()");


    // --- RETURN -----------------------------------------------
    cudaFree( *input );
    *input = tmp_output;
    // ----------------------------------------------------------
	
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_pad()
//
// KERNELS INVOKED:
//   kernel_pad()
//
// AUTHOR: James Shackleford
//   DATE: 10 September, 2009
////////////////////////////////////////////////////////////////////////////////
extern "C" void CUDA_pad(
    float** input,
    int* vol_dims,
    int* tile_dims)
{

    // --- CALCULATE THINGS NEEDED BY THIS STUB -----------------
    int3 vol_dim;
    vol_dim.x = vol_dims[0];
    vol_dim.y = vol_dims[1];
    vol_dim.z = vol_dims[2];

    int3 tile_dim;
    tile_dim.x = tile_dims[0];
    tile_dim.y = tile_dims[1];
    tile_dim.z = tile_dims[2];

    int num_voxels = vol_dim.x * vol_dim.y * vol_dim.z;

    int4 num_tiles;
    num_tiles.x = (vol_dim.x+tile_dim.x-1) / tile_dim.x;
    num_tiles.y = (vol_dim.y+tile_dim.y-1) / tile_dim.y;
    num_tiles.z = (vol_dim.z+tile_dim.z-1) / tile_dim.z;
    num_tiles.w = num_tiles.x * num_tiles.y * num_tiles.z;

    int tile_padding = 32 - ((tile_dim.x * tile_dim.y * tile_dim.z) % 32);
    int tile_bytes = (tile_dim.x * tile_dim.y * tile_dim.z);

    int output_size = (tile_bytes + tile_padding) * num_tiles.w;
    // ----------------------------------------------------------



    // --- ALLOCATE GPU GLOBAL MEMORY FOR OUTPUT ----------------
    float* tmp_output;
    cudaMalloc((void**)&tmp_output, output_size*sizeof(float));
    cudaMemset(tmp_output, 0, output_size*sizeof(float));
    // ----------------------------------------------------------

    // --- INITIALIZE GRID --------------------------------------
    int i;
    int warps_per_block = 4;
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = (num_voxels+threads_per_block-1) / threads_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_pad() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    kernel_pad<<<dimGrid, dimBlock>>>(
	*input,
	tmp_output,
	vol_dim,
	tile_dim
    );
    // ----------------------------------------------------------

    cudaThreadSynchronize();
    checkCUDAError("[Kernel Panic!] kernel_pad()");


    // --- RETURN -----------------------------------------------
    cudaFree( *input );
    *input = tmp_output;
    // ----------------------------------------------------------
	
}


//////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_bspline_mse_score_dc_dv()
//
// KERNELS INVOKED:
//   kernel_bspline_mse_2_reduce()
//
// AUTHOR: James Shackleford
//   DATE: 19 August, 2009
//////////////////////////////////////////////////////////////////////////////
extern "C" void CUDA_bspline_mse_score_dc_dv (
    Dev_Pointers_Bspline* dev_ptrs,
    BSPLINE_Xform* bxf,
    Volume* fixed,
    Volume* moving)
{
    // JAS 10.15.2009
    // TODO: A structure similar to the BSpline_Xform
    //       that uses float3s needs to be used here
    //       to clean up the code.  Perhaps...

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

    // Fixed image header
    int3 fix_dim;
    fix_dim.x = fixed->dim[0]; 
    fix_dim.y = fixed->dim[1];
    fix_dim.z = fixed->dim[2];

    float3 fix_origin;		
    fix_origin.x = (float) bxf->img_origin[0];
    fix_origin.y = (float) bxf->img_origin[1];
    fix_origin.z = (float) bxf->img_origin[2];

    float3 fix_spacing;
    fix_spacing.x = (float) bxf->img_spacing[0];
    fix_spacing.y = (float) bxf->img_spacing[1];
    fix_spacing.z = (float) bxf->img_spacing[2];

    // Moving image header
    int3 mov_dim;		
    mov_dim.x = moving->dim[0]; 
    mov_dim.y = moving->dim[1];
    mov_dim.z = moving->dim[2];

    float3 mov_origin;
    mov_origin.x = (float) moving->offset[0];
    mov_origin.y = (float) moving->offset[1];
    mov_origin.z = (float) moving->offset[2];

    float3 mov_spacing;
    mov_spacing.x = (float) moving->pix_spacing[0];
    mov_spacing.y = (float) moving->pix_spacing[1];
    mov_spacing.z = (float) moving->pix_spacing[2];

    // Dimension of ROI (in vox)
    int3 roi_dim;           
    roi_dim.x = bxf->roi_dim[0];	
    roi_dim.y = bxf->roi_dim[1];
    roi_dim.z = bxf->roi_dim[2];

    // Position of first vox in ROI (in vox)
    int3 roi_offset;        
    roi_offset.x = bxf->roi_offset[0];
    roi_offset.y = bxf->roi_offset[1];
    roi_offset.z = bxf->roi_offset[2];

    // Number of voxels per region
    int3 vox_per_rgn;		
    vox_per_rgn.x = bxf->vox_per_rgn[0];
    vox_per_rgn.y = bxf->vox_per_rgn[1];
    vox_per_rgn.z = bxf->vox_per_rgn[2];

    // --- INITIALIZE GRID ---
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 128;
    int num_threads = fixed->npix;
    int sqrt_num_blocks;
    int num_blocks;
    int smemSize;
    int found_flag = 0;

    // Search for a valid execution configuration
    // for the required # of blocks.
    for (threads_per_block = 192; threads_per_block > 32; threads_per_block -= 32) {
	num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
	smemSize = 12 * sizeof(float) * threads_per_block;
	sqrt_num_blocks = (int)sqrt((float)num_blocks);

	for (i = sqrt_num_blocks; i < 65535; i++) {
	    if (num_blocks % i == 0) {
		Grid_x = i;
		Grid_y = num_blocks / Grid_x;
		found_flag = 1;
		break;
	    }
	}

	if (found_flag == 1) {
	    break;
	}
    }

    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable bspline_cuda_score_j_mse_kernel1() configuration!\n");
	exit(0);
    } else {
#if defined (commentout)
	printf ("Grid [%i,%i], %d threads_per_block.\n", 
	    Grid_x, Grid_y, threads_per_block);
#endif
    }

    dim3 dimGrid1(Grid_x, Grid_y, 1);
    dim3 dimBlock1(threads_per_block, 1, 1);

    // --- BEGIN KERNEL EXECUTION ---
    //	cudaEvent_t start, stop;
    //	float time;

    //	cudaEventCreate(&start);
    //	cudaEventCreate(&stop);

    //	cudaEventRecord (start, 0);	

    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_x");

    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_y");

    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);
    checkCUDAError("cudaMemset(): dev_ptrs->dc_dv_z");

    int tile_padding = 64 - 
	((vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z) % 64);

    /* GCS ??? */
    //if (tile_padding == 64) tile_padding = 0;
    printf ("tile_padding = %d\n", tile_padding);

    bspline_cuda_score_j_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>(
	dev_ptrs->dc_dv_x,	// Addr of dc_dv_x on GPU
	dev_ptrs->dc_dv_y,	// Addr of dc_dv_y on GPU
	dev_ptrs->dc_dv_z,	// Addr of dc_dv_z on GPU
	dev_ptrs->score,	// Addr of score on GPU
	dev_ptrs->coeff,	// Addr of coeff on GPU
	dev_ptrs->fixed_image,	// Addr of fixed_image on GPU
	dev_ptrs->moving_image,	// Addr of moving_image on GPU
	dev_ptrs->moving_grad,  // Addr of moving_grad on GPU
	fix_dim,                // Size of fixed image (vox)
	fix_origin,             // Origin of fixed image (mm)
	fix_spacing,            // Spacing of fixed image (mm)
	mov_dim,                // Size of moving image (vox)
	mov_origin,             // Origin of moving image (mm)
	mov_spacing,            // Spacing of moving image (mm)
	roi_dim,		// Region of Intrest Dimenions
	roi_offset,		// Region of Intrest Offset
	vox_per_rgn,		// Voxels per Region
	rdims,                  // # of regions in (x,y,z)
	cdims,                  // # of control points in (x,y,z)
	tile_padding,
	dev_ptrs->skipped);

    //	cudaEventRecord (stop, 0);	
    //	cudaEventSynchronize (stop);
    //	cudaEventElapsedTime (&time, start, stop);
    //	cudaEventDestroy (start);
    //	cudaEventDestroy (stop);
    //	printf("\n[%f ms] MSE & dc_dv\n", time);
}


//////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_bspline_mse_2_condense_64_texfetch()
//
// KERNELS INVOKED:
//   kernel_bspline_mse_2_condense_64()
//
// AUTHOR: James Shackleford
//   DATE: September 16th, 2009
//////////////////////////////////////////////////////////////////////////////
void CUDA_bspline_mse_2_condense_64_texfetch (
    Dev_Pointers_Bspline* dev_ptrs,
    int* vox_per_rgn,
    int num_tiles)
{
    int4 vox_per_region;
    vox_per_region.x = vox_per_rgn[0];
    vox_per_region.y = vox_per_rgn[1];
    vox_per_region.z = vox_per_rgn[2];
    vox_per_region.w = vox_per_region.x * vox_per_region.y * vox_per_region.z;

    int pad = 64 - (vox_per_region.w % 64);

    vox_per_region.w += pad;

    // --- INITIALIZE GRID --------------------------------------
    // LAUNCH KERNEL WITH # THREAD BLOCKS = # TILES
    // WITH # WARPS PER THREAD BLOCK = 1
    int i;
    int warps_per_block = 2;
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = num_tiles;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_bspline_mse_2_condense_64_texfetch() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    //	int smemSize = 384*sizeof(float);
    int smemSize = 576*sizeof(float);
    // ----------------------------------------------------------


    //	printf("\nLaunching CONDENSE with %i threadblocks\n", num_blocks);

    kernel_bspline_mse_2_condense_64_texfetch<<<dimGrid, dimBlock, smemSize>>>(
	dev_ptrs->cond_x,		// Return: condensed dc_dv_x values
	dev_ptrs->cond_y,		// Return: condensed dc_dv_y values
	dev_ptrs->cond_z,		// Return: condensed dc_dv_z values
	dev_ptrs->dc_dv_x,		// Input : dc_dv_x values
	dev_ptrs->dc_dv_y,		// Input : dc_dv_y values
	dev_ptrs->dc_dv_z,		// Input : dc_dv_z values
	dev_ptrs->LUT_Offsets,		// Input : tile offsets
	dev_ptrs->LUT_Knot,		// Input : linear knot indicies
	pad,				// Input : amount of tile padding
	vox_per_region,			// Input : dims of tiles
	(float)1/6);			// Input : GPU Division is slow
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_bspline_mse_2_condense_64()
//
// KERNELS INVOKED:
//   kernel_bspline_mse_2_condense_64()
//
// AUTHOR: James Shackleford
//   DATE: September 16th, 2009
////////////////////////////////////////////////////////////////////////////////
void CUDA_bspline_mse_2_condense_64(
    Dev_Pointers_Bspline* dev_ptrs,
    int* vox_per_rgn,
    int num_tiles)
{
    int4 vox_per_region;
    vox_per_region.x = vox_per_rgn[0];
    vox_per_region.y = vox_per_rgn[1];
    vox_per_region.z = vox_per_rgn[2];
    vox_per_region.w = vox_per_region.x * vox_per_region.y * vox_per_region.z;

    int pad = 64 - (vox_per_region.w % 64);

    vox_per_region.w += pad;

    // --- INITIALIZE GRID --------------------------------------
    // LAUNCH KERNEL WITH # THREAD BLOCKS = # TILES
    // WITH # WARPS PER THREAD BLOCK = 1
    int i;
    int warps_per_block = 2;
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = num_tiles;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_bspline_mse_2_condense_64() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    int smemSize = 384*sizeof(float);
    //	int smemSize = 288*sizeof(float);
    // ----------------------------------------------------------


    //	printf("\nLaunching CONDENSE with %i threadblocks\n", num_blocks);

    kernel_bspline_mse_2_condense_64<<<dimGrid, dimBlock, smemSize>>>(
	dev_ptrs->cond_x,		// Return: condensed dc_dv_x values
	dev_ptrs->cond_y,		// Return: condensed dc_dv_y values
	dev_ptrs->cond_z,		// Return: condensed dc_dv_z values
	dev_ptrs->dc_dv_x,		// Input : dc_dv_x values
	dev_ptrs->dc_dv_y,		// Input : dc_dv_y values
	dev_ptrs->dc_dv_z,		// Input : dc_dv_z values
	dev_ptrs->LUT_Offsets,		// Input : tile offsets
	dev_ptrs->LUT_Knot,		// Input : linear knot indicies
	pad,				// Input : amount of tile padding
	vox_per_region,			// Input : dims of tiles
	(float)1/6);			// Input : GPU Division is slow
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_bspline_mse_2_condense()
//
// KERNELS INVOKED:
//   kernel_bspline_mse_2_condense()
//
// AUTHOR: James Shackleford
//   DATE: 19 August, 2009
////////////////////////////////////////////////////////////////////////////////
void CUDA_bspline_mse_2_condense(
    Dev_Pointers_Bspline* dev_ptrs,
    int* vox_per_rgn,
    int num_tiles)
{
    int4 vox_per_region;
    vox_per_region.x = vox_per_rgn[0];
    vox_per_region.y = vox_per_rgn[1];
    vox_per_region.z = vox_per_rgn[2];
    vox_per_region.w = vox_per_region.x * vox_per_region.y * vox_per_region.z;

    int pad = 32 - (vox_per_region.w % 32);


    // --- INITIALIZE GRID --------------------------------------
    // LAUNCH KERNEL WITH # THREAD BLOCKS = # TILES
    // WITH # WARPS PER THREAD BLOCK = 1
    int i;
    int warps_per_block = 1;
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = (num_tiles+warps_per_block-1) / warps_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_bspline_mse_2_condense() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    int smemSize = 384*sizeof(float);
    //	int smemSize = 288*sizeof(float);
    // ----------------------------------------------------------


    //	printf("\nLaunching CONDENSE with %i threadblocks\n", num_blocks);

    kernel_bspline_mse_2_condense<<<dimGrid, dimBlock, smemSize>>>(
	dev_ptrs->cond_x,		// Return: condensed dc_dv_x values
	dev_ptrs->cond_y,		// Return: condensed dc_dv_y values
	dev_ptrs->cond_z,		// Return: condensed dc_dv_z values
	dev_ptrs->dc_dv_x,		// Input : dc_dv_x values
	dev_ptrs->dc_dv_y,		// Input : dc_dv_y values
	dev_ptrs->dc_dv_z,		// Input : dc_dv_z values
	dev_ptrs->LUT_Offsets,		// Input : tile offsets
	dev_ptrs->LUT_Knot,		// Input : linear knot indicies
	pad,				// Input : amount of tile padding
	vox_per_region,			// Input : dims of tiles
	(float)1/6);			// Input : GPU Division is slow
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_bspline_mse_2_reduce()
//
// KERNELS INVOKED:
//   kernel_bspline_mse_2_reduce()
//
// AUTHOR: James Shackleford
//   DATE: 19 August, 2009
////////////////////////////////////////////////////////////////////////////////
extern "C" void CUDA_bspline_mse_2_reduce (
    Dev_Pointers_Bspline* dev_ptrs,
    int num_knots)
{

    // --- INITIALIZE GRID --------------------------------------
    // LAUNCH KERNEL WITH # THREAD BLOCKS = # KNOTS / 32
    // WITH # WARPS PER THREAD BLOCK = 2
    int i;
    int warps_per_block = 2;
    int knots_per_block = 1;
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = (num_knots+knots_per_block-1) / knots_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
	if (num_blocks % i == 0)
	{
	    Grid_x = i;
	    Grid_y = num_blocks / Grid_x;
	    break;
	}
    }
    // *****


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
	// If this happens we should consider falling back to a
	// CPU implementation, using a different CUDA algorithm,
	// or padding the input dc_dv stream to work with this
	// CUDA algorithm.
	printf("\n[ERROR] Unable to find suitable CUDA_bspline_mse_2_reduce() configuration!\n");
	exit(0);
    } else {
	//		printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    int smemSize = 195*sizeof(float);
    // ----------------------------------------------------------


    //	printf("\nLaunching REDUCE with %i threadblocks\n", num_blocks);

    kernel_bspline_mse_2_reduce<<<dimGrid, dimBlock, smemSize>>>(
	dev_ptrs->grad,		// Return: interleaved dc_dp values
	dev_ptrs->cond_x,	// Input : condensed dc_dv_x values
	dev_ptrs->cond_y,	// Input : condensed dc_dv_y values
	dev_ptrs->cond_z);	// Input : condensed dc_dv_z values

}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// FUNCTION: CPU_obtain_spline_basis_function()
//
// AUTHOR: James Shackleford
// DATE  : 09.04.2009
////////////////////////////////////////////////////////////////////////////////
float CPU_obtain_spline_basis_function( int t_idx, 
    int vox_idx, 
    int vox_per_rgn)
{
								
    float i = (float)vox_idx / vox_per_rgn;
    float C;
						
    switch(t_idx) {
    case 0:
	C = (1.0/6.0) * (- 1.0 * i*i*i + 3.0 * i*i - 3.0 * i + 1.0);
	break;
    case 1:
	C = (1.0/6.0) * (+ 3.0 * i*i*i - 6.0 * i*i           + 4.0);
	break;
    case 2:
	C = (1.0/6.0) * (- 3.0 * i*i*i + 3.0 * i*i + 3.0 * i + 1.0);
	break;
    case 3:
	C = (1.0/6.0) * (+ 1.0 * i*i*i);
	break;
    default:
	C = 0.0;
	break;
    }
    return C;
}
////////////////////////////////////////////////////////////////////////////////



/***********************************************************************
 * checkCUDAError
 *
 * If a CUDA error is detected, this function gets the error message
 * and displays it to the user.
 ***********************************************************************/
void checkCUDAError (const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err) 
    {
	printf("CUDA Error -- %s: %s.\n", msg, cudaGetErrorString(err));
	exit(-1);
    } 
}
