/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if defined (_WIN32)
#include <windows.h>
#endif

#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_cuda.h"
#include "bspline_cuda_kernels.h"
#include "cuda_utils.h"
#include "mha_io.h"
#include "volume.h"

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

#define USE_TEXTURES 1      // Textures Enabled
//#define USE_TEXTURES 0    // Textures Disabled

#if defined (USE_TEXTURES)
#define TEX_REF(array,index) \
    (tex1Dfetch(tex_ ## array, index))
#else
#define TEX_REF(array,index) \
    (array[index])
#endif

#define GRID_LIMIT_X 65535
#define GRID_LIMIT_Y 65535
////////////////////////////////////////////////////////////

// Uncomment to include profiling code for MSE CUDA flavor J
//#define PROFILE_J

typedef struct gpu_bspline_data GPU_Bspline_Data;
struct gpu_bspline_data
{
    // bxf items
    int3 rdims;         
    int3 cdims;
    float3 img_origin;      
    float3 img_spacing;
    int3 roi_dim;           
    int3 roi_offset;        
    int3 vox_per_rgn;       

    // fixed volume items
    int3 fix_dim;

    // moving volume items
    int3 mov_dim;       
    float3 mov_offset;
    float3 mov_spacing;
};


// Constructs the GPU Bspline Data structure
void
build_gbd (
    GPU_Bspline_Data* gbd,
    Bspline_xform* bxf,
    Volume* fixed,
    Volume* moving)
{
    if (bxf != NULL) {
        // populate bxf entries
        memcpy (&gbd->rdims, bxf->rdims, 3*sizeof(int));
        memcpy (&gbd->cdims, bxf->cdims, 3*sizeof(int));
        memcpy (&gbd->img_origin, bxf->img_origin, 3*sizeof(float));
        memcpy (&gbd->img_spacing, bxf->img_spacing, 3*sizeof(float));
        memcpy (&gbd->roi_dim, bxf->roi_dim, 3*sizeof(int));
        memcpy (&gbd->roi_offset, bxf->roi_offset, 3*sizeof(int));
        memcpy (&gbd->vox_per_rgn, bxf->vox_per_rgn, 3*sizeof(int));
    }

    if (fixed != NULL) {
        // populate fixed volume entries
        memcpy (&gbd->fix_dim, fixed->dim, 3*sizeof(int));
    }

    if (moving != NULL) {
        // populate moving volume entries
        memcpy (&gbd->mov_dim, moving->dim, 3*sizeof(int));
        memcpy (&gbd->mov_offset, moving->offset, 3*sizeof(float));
        memcpy (&gbd->mov_spacing, moving->pix_spacing, 3*sizeof(float));
    }
    
}


// Builds execution configurations for kernels that
// assign one thread per element (1tpe).
int
build_exec_conf_1tpe (
    dim3 *dimGrid,          // OUTPUT: Grid  dimensions
    dim3 *dimBlock,         // OUTPUT: Block dimensions
    int num_threads,        // INPUT: Total # of threads
    int threads_per_block,  // INPUT: Threads per block
    bool negotiate          // INPUT: Is threads per block negotiable?
)
{
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int sqrt_num_blocks;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    if (negotiate) {
        int found_flag = 0;
        int j = 0;

        // Search for a valid execution configuration for the required # of blocks.
        // Block size has been specified as changable.  This helps if the
        // number of blocks required is a prime number > 65535.  Changing the
        // # of threads per block will change the # of blocks... which hopefully
        // won't be prime again.
        for (j = threads_per_block; j > 32; j -= 32) {
            num_blocks = (num_threads + j - 1) / j;
            sqrt_num_blocks = (int)sqrt((float)num_blocks);

            for (i = sqrt_num_blocks; i < GRID_LIMIT_X; i++) {
                if (num_blocks % i == 0) {
                    Grid_x = i;
                    Grid_y = num_blocks / Grid_x;
                    found_flag = 1;
                    break;
                }
            }

            if (found_flag == 1) {
                threads_per_block = j;
                break;
            }
        }

    } else {

        // Search for a valid execution configuration for the required # of blocks.
        // The calling algorithm has specifed that # of threads per block
        // is non negotiable.
        sqrt_num_blocks = (int)sqrt((float)num_blocks);

        for (i = sqrt_num_blocks; i < GRID_LIMIT_X; i++) {
            if (num_blocks % i == 0) {
                Grid_x = i;
                Grid_y = num_blocks / Grid_x;
                break;
            }
        }
    }



    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
        printf ("\n");
        printf ("[GPU KERNEL PANIC] Unable to find suitable execution configuration!");
        printf ("Terminating...\n");
        exit (0);
    } else {
        // callback function could be added
        // to arguments and called here if you need
        // to do something fancy upon success.
#if VERBOSE
        printf ("Grid [%i,%i], %d threads_per_block.\n", 
            Grid_x, Grid_y, threads_per_block);
#endif
    }

    // Pass configuration back by reference
    dimGrid->x = Grid_x;
    dimGrid->y = Grid_y;
    dimGrid->z = 1;

    dimBlock->x = threads_per_block;
    dimBlock->y = 1;
    dimBlock->z = 1;

    // Return the # of blocks we decided on just
    // in case we need it later to allocate shared memory, etc.
    return num_blocks;
}

// Builds execution configurations for kernels that
// assign one block per element (1bpe).
void
build_exec_conf_1bpe (
    dim3 *dimGrid,          // OUTPUT: Grid  dimensions
    dim3 *dimBlock,         // OUTPUT: Block dimensions
    int num_blocks,         // INPUT: Number of blocks
    int threads_per_block)  // INPUT: Threads per block
{
    int i;
    int Grid_x = 0;
    int Grid_y = 0;

    // Search for a valid execution configuration for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++) {
        if (num_blocks % i == 0) {
            Grid_x = i;
            Grid_y = num_blocks / Grid_x;
            break;
        }
    }


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
        printf ("\n");
        printf ("[GPU KERNEL PANIC] Unable to find suitable execution configuration!");
        printf ("Terminating...\n");
        exit (0);
    } else {
        // callback function could be added
        // to arguments and called here if you need
        // to do something fancy upon success.
#if VERBOSE
        printf ("Grid [%i,%i], %d threads_per_block.\n", 
            Grid_x, Grid_y, threads_per_block);
#endif
    }

    // Pass configuration back by reference
    dimGrid->x = Grid_x;
    dimGrid->y = Grid_y;
    dimGrid->z = 1;

    dimBlock->x = threads_per_block;
    dimBlock->y = 1;
    dimBlock->z = 1;

}
    



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

    if (threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z)) {
        dx[threadIdxInGrid] = (float)threadIdxInGrid;
        dy[threadIdxInGrid] = (float)threadIdxInGrid;
        dz[threadIdxInGrid] = (float)threadIdxInGrid;
    }
}

extern "C" void
bspline_cuda_init_MI_a (
    Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
    BSPLINE_Parms* parms)
{
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;

    printf ("Allocating GPU Memory");

    // input volumes
    dev_ptrs->fixed_image_size = fixed->npix * sizeof(float);
    dev_ptrs->moving_image_size = moving->npix * sizeof(float);
    dev_ptrs->moving_grad_size = moving_grad->npix * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->fixed_image, dev_ptrs->fixed_image_size);
        cuda_utils_check_error ("Failed to allocate memory for fixed image");
    cudaMalloc ((void**)&dev_ptrs->moving_image, dev_ptrs->moving_image_size);
        cuda_utils_check_error ("Failed to allocate memory for moving image");
    cudaMalloc ((void**)&dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
        cuda_utils_check_error ("Failed to allocate memory for moving grad");
    cudaMemcpy (dev_ptrs->fixed_image, fixed->img, dev_ptrs->fixed_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy (dev_ptrs->moving_image, moving->img, dev_ptrs->moving_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy (dev_ptrs->moving_grad, moving_grad->img, dev_ptrs->moving_grad_size, cudaMemcpyHostToDevice);

    // skipped voxels
    dev_ptrs->skipped_size = sizeof(float) * fixed->npix;
    cudaMalloc ((void**)&dev_ptrs->skipped, dev_ptrs->skipped_size);
    cudaMemset (dev_ptrs->skipped, 0, dev_ptrs->skipped_size);

#if defined (commentout)
    // segmented histograms
    int num_blocks = (fixed->npix + 31) / 32;
    dev_ptrs->f_hist_seg_size = mi_hist->fixed.bins * 2*num_blocks * sizeof(float);
    dev_ptrs->m_hist_seg_size = mi_hist->moving.bins * num_blocks * sizeof(float);
    dev_ptrs->j_hist_seg_size = mi_hist->fixed.bins * num_blocks * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->f_hist_seg, dev_ptrs->f_hist_seg_size);
        cuda_utils_check_error ("Failed to allocate memory for f_hist_seg");
    cudaMalloc ((void**)&dev_ptrs->m_hist_seg, dev_ptrs->m_hist_seg_size);
        cuda_utils_check_error ("Failed to allocate memory for m_hist_seg");
    cudaMalloc ((void**)&dev_ptrs->j_hist_seg, dev_ptrs->j_hist_seg_size);
        cuda_utils_check_error ("Failed to allocate memory for j_hist_seg");
#endif


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
        cuda_utils_check_error("Failed to allocate memory for q_LUT");

    cudaMemcpy(dev_ptrs->q_lut, bxf->q_lut, dev_ptrs->q_lut_size, cudaMemcpyHostToDevice);
        cuda_utils_check_error("Failed to copy multiplier q_LUT to GPU");

    cudaBindTexture(0, tex_q_lut, dev_ptrs->q_lut, dev_ptrs->q_lut_size);
        cuda_utils_check_error("Failed to bind tex_q_lut to texture");

    // Copy the index LUT to the GPU.
    dev_ptrs->c_lut_size = sizeof(int) 
    * bxf->rdims[0] 
    * bxf->rdims[1] 
    * bxf->rdims[2] 
    * 64;

    cudaMalloc((void**)&dev_ptrs->c_lut, dev_ptrs->c_lut_size);
        cuda_utils_check_error("Failed to allocate memory for c_LUT");
    cudaMemcpy(dev_ptrs->c_lut, bxf->c_lut, dev_ptrs->c_lut_size, cudaMemcpyHostToDevice);
        cuda_utils_check_error("Failed to copy index c_LUT to GPU");
    cudaBindTexture(0, tex_c_lut, dev_ptrs->c_lut, dev_ptrs->c_lut_size);
        cuda_utils_check_error("Failed to bind tex_c_lut to texture");

    // coefficients
    dev_ptrs->coeff_size = sizeof(float) * bxf->num_coeff;
    cudaMalloc((void**)&dev_ptrs->coeff, dev_ptrs->coeff_size);
        cuda_utils_check_error("Failed to allocate memory for dev_ptrs->coeff");
    cudaMemset(dev_ptrs->coeff, 0, dev_ptrs->coeff_size);
    cudaBindTexture(0, tex_coeff, dev_ptrs->coeff, dev_ptrs->coeff_size);
        cuda_utils_check_error("Failed to bind dev_ptrs->coeff to texture reference!");

    // score
    dev_ptrs->score_size = sizeof(float) * fixed->npix;
    dev_ptrs->skipped_size = sizeof(float) * fixed->npix;
    cudaMalloc((void**)&dev_ptrs->score, dev_ptrs->score_size);
    
    // grad
    dev_ptrs->grad_size = sizeof(float) * bxf->num_coeff;
    cudaMalloc((void**)&dev_ptrs->grad, dev_ptrs->grad_size);

    // dc_dv_x, dc_dv_y, and dc_dv_z
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

    cudaMalloc((void**)&dev_ptrs->dc_dv_x, dev_ptrs->dc_dv_x_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_x");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->dc_dv_y, dev_ptrs->dc_dv_y_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_y");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->dc_dv_z, dev_ptrs->dc_dv_z_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_z");
    printf(".");

    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);


    // cond_x y and z
    dev_ptrs->cond_x_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_y_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_z_size = 64*bxf->num_knots*sizeof(float);

    cudaMalloc((void**)&dev_ptrs->cond_x, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_x");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->cond_y, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_y");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->cond_z, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_z");
    printf(".");

    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_z");


    // tile offset lut  (needed by condense_64_texfetch)
    int* offsets = calc_offsets(bxf->vox_per_rgn, bxf->cdims);

    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    dev_ptrs->LUT_Offsets_size = num_tiles*sizeof(int);

    cudaMalloc((void**)&dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->LUT_Offsets");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Offsets, offsets, dev_ptrs->LUT_Offsets_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("cudaMemcpy(): offsets --> dev_ptrs->LUT_Offsets");
    cudaBindTexture(0, tex_LUT_Offsets, dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);

    free (offsets);

    // knot lut (needed by condense_64_texfetch)
    dev_ptrs->LUT_Knot_size = 64*num_tiles*sizeof(int);

    int* local_set_of_64 = (int*)malloc(64*sizeof(int));
    int* LUT_Knot = (int*)malloc(dev_ptrs->LUT_Knot_size);

    int i,j;
    for (i = 0; i < num_tiles; i++) {
        find_knots(local_set_of_64, i, bxf->rdims, bxf->cdims);
        for (j = 0; j < 64; j++) {
            LUT_Knot[64*i + j] = local_set_of_64[j];
        }
    }

    cudaMalloc((void**)&dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->LUT_Knot");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Knot, LUT_Knot, dev_ptrs->LUT_Knot_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("cudaMemcpy(): LUT_Knot --> dev_ptrs->LUT_Knot");

    free (local_set_of_64);
    free (LUT_Knot);


    // --- GENERATE B-SPLINE LOOK UP TABLE ----------------------
    dev_ptrs->LUT_Bspline_x_size = 4*bxf->vox_per_rgn[0]* sizeof(float);
    dev_ptrs->LUT_Bspline_y_size = 4*bxf->vox_per_rgn[1]* sizeof(float);
    dev_ptrs->LUT_Bspline_z_size = 4*bxf->vox_per_rgn[2]* sizeof(float);
    float* LUT_Bspline_x = (float*)malloc(dev_ptrs->LUT_Bspline_x_size);
    float* LUT_Bspline_y = (float*)malloc(dev_ptrs->LUT_Bspline_y_size);
    float* LUT_Bspline_z = (float*)malloc(dev_ptrs->LUT_Bspline_z_size);

    for (j = 0; j < 4; j++) {
        for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
            LUT_Bspline_x[j*bxf->vox_per_rgn[0] + i] = CPU_obtain_spline_basis_function (j, i, bxf->vox_per_rgn[0]);
        }

        for (i = 0; i < bxf->vox_per_rgn[1]; i++) {
            LUT_Bspline_y[j*bxf->vox_per_rgn[1] + i] = CPU_obtain_spline_basis_function (j, i, bxf->vox_per_rgn[1]);
        }

        for (i = 0; i < bxf->vox_per_rgn[2]; i++) {
            LUT_Bspline_z[j*bxf->vox_per_rgn[2] + i] = CPU_obtain_spline_basis_function (j, i, bxf->vox_per_rgn[2]);
        }
    }
    
    cudaMalloc((void**)&dev_ptrs->LUT_Bspline_x, dev_ptrs->LUT_Bspline_x_size);
    cudaMalloc((void**)&dev_ptrs->LUT_Bspline_y, dev_ptrs->LUT_Bspline_y_size);
    cudaMalloc((void**)&dev_ptrs->LUT_Bspline_z, dev_ptrs->LUT_Bspline_z_size);

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



    printf (" done.\n");

}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_initialize_j()
// 
// Initialize the GPU to execute bspline_cuda_score_j_mse().
//
// AUTHOR: James Shackleford
// DATE  : September 17, 2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_cuda_initialize_j(Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
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
    cuda_utils_check_error ("Failed to allocate memory for fixed image");
    printf(".");


    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy (dev_ptrs->fixed_image, fixed->img, 
	dev_ptrs->fixed_image_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error ("Failed to copy fixed image to GPU");
    printf(".");


    // Bind this to a texture reference
    cudaBindTexture (0, tex_fixed_image, dev_ptrs->fixed_image, 
	dev_ptrs->fixed_image_size);
    cuda_utils_check_error ("Failed to bind dev_ptrs->fixed_image to texture reference!");
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
    cuda_utils_check_error("Failed to allocate memory for moving image");
    printf(".");
    
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy( dev_ptrs->moving_image, moving->img, dev_ptrs->moving_image_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("Failed to copy moving image to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_image, dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->moving_image to texture reference!");
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
    cuda_utils_check_error("Failed to allocate memory for moving grad");
    printf(".");
    
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    // (Note the pointer dereference)
    cudaMemcpy( dev_ptrs->moving_grad, moving_grad->img, dev_ptrs->moving_grad_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("Failed to copy moving grad to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_grad, dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->moving_image to texture reference!");
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
    cuda_utils_check_error("Failed to allocate memory for dev_ptrs->coeff");
    printf(".");


    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->coeff, 0, dev_ptrs->coeff_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_coeff, dev_ptrs->coeff, dev_ptrs->coeff_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->coeff to texture reference!");
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
    cuda_utils_check_error("Failed to bind dev_ptrs->grad to texture reference!");
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
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_x");
    printf(".");
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->dc_dv_y_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->dc_dv_y, dev_ptrs->dc_dv_y_size);
    GPU_Memory_Bytes += dev_ptrs->dc_dv_y_size;
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_y");
    printf(".");
#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->dc_dv_z_size,
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->dc_dv_z, dev_ptrs->dc_dv_z_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_z");
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

    //  int vox_per_tile = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    //  int pad = 64 - (vox_per_tile % 64);

    dev_ptrs->LUT_Offsets_size = num_tiles*sizeof(int);

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->LUT_Offsets_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->LUT_Offsets");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Offsets, offsets, dev_ptrs->LUT_Offsets_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("cudaMemcpy(): offsets --> dev_ptrs->LUT_Offsets");
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
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->LUT_Knot");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Knot, LUT_Knot, dev_ptrs->LUT_Knot_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("cudaMemcpy(): LUT_Knot --> dev_ptrs->LUT_Knot");

    //  cudaBindTexture(0, tex_LUT_Knot, dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    //  cuda_utils_check_error("cudaBindTexture(): dev_ptrs->LUT_Knot");

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
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_x");
    printf(".");

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->cond_y_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->cond_y, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_y");
    printf(".");

#if VERBOSE
    printf ("Trying to allocate %lu (%lu already allocated)\n",
	(long unsigned) dev_ptrs->cond_z_size, 
	GPU_Memory_Bytes);
#endif
    cudaMalloc((void**)&dev_ptrs->cond_z, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_z");
    printf(".");

    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_z");

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
void
bspline_cuda_initialize_i(Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
    BSPLINE_Parms* parms)
{
    // Keep track of how much memory we allocated
    // in the GPU global memory.
    int GPU_Memory_Bytes = 0;
    //  int temp;

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
    cuda_utils_check_error("Failed to allocate memory for fixed image");
    printf(".");


    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy( dev_ptrs->fixed_image, fixed->img, dev_ptrs->fixed_image_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("Failed to copy fixed image to GPU");
    printf(".");


    // Bind this to a texture reference
    cudaBindTexture(0, tex_fixed_image, dev_ptrs->fixed_image, dev_ptrs->fixed_image_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->fixed_image to texture reference!");
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
    cuda_utils_check_error("Failed to allocate memory for moving image");
    printf(".");
    
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    cudaMemcpy( dev_ptrs->moving_image, moving->img, dev_ptrs->moving_image_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("Failed to copy moving image to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_image, dev_ptrs->moving_image, dev_ptrs->moving_image_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->moving_image to texture reference!");
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
    cuda_utils_check_error("Failed to allocate memory for moving grad");
    printf(".");
    
    // Populate the newly allocated global GPU memory
    // with the voxel data from our fixed volume.
    // (Note the pointer dereference)
    cudaMemcpy( dev_ptrs->moving_grad, moving_grad->img, dev_ptrs->moving_grad_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("Failed to copy moving grad to GPU");
    printf(".");

    // Bind this to a texture reference
    cudaBindTexture(0, tex_moving_grad, dev_ptrs->moving_grad, dev_ptrs->moving_grad_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->moving_image to texture reference!");
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
    cuda_utils_check_error("Failed to allocate memory for dev_ptrs->coeff");
    printf(".");


    // Cuda does not automatically zero out malloc()ed blocks
    // of memory that have been allocated in GPU global
    // memory.  So, we zero them out ourselves.
    cudaMemset(dev_ptrs->coeff, 0, dev_ptrs->coeff_size);

    // Bind this to a texture reference
    cudaBindTexture(0, tex_coeff, dev_ptrs->coeff, dev_ptrs->coeff_size);
    cuda_utils_check_error("Failed to bind dev_ptrs->coeff to texture reference!");
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
    cuda_utils_check_error("Failed to bind dev_ptrs->dc_dv to texture reference!");
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
    cuda_utils_check_error("Failed to bind dev_ptrs->grad to texture reference!");
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
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_x");
    printf(".");
    cudaMalloc((void**)&dev_ptrs->dc_dv_y, dev_ptrs->dc_dv_y_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_y");
    printf(".");
    cudaMalloc((void**)&dev_ptrs->dc_dv_z, dev_ptrs->dc_dv_z_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->dc_dv_z");
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

    //  int vox_per_tile = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    //  int pad = 64 - (vox_per_tile % 64);

    dev_ptrs->LUT_Offsets_size = num_tiles*sizeof(int);

    cudaMalloc((void**)&dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->LUT_Offsets");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Offsets, offsets, dev_ptrs->LUT_Offsets_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("cudaMemcpy(): offsets --> dev_ptrs->LUT_Offsets");
    cudaBindTexture(0, tex_LUT_Offsets, dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);

    free (offsets);

    GPU_Memory_Bytes += dev_ptrs->LUT_Offsets_size;
    // ----------------------------------------------------------

    // --- ALLOCATE KNOT LUT IN GPU GLOBAL ----------------------
    dev_ptrs->LUT_Knot_size = 64*num_tiles*sizeof(int);

    int* local_set_of_64 = (int*)malloc(64*sizeof(int));
    int* LUT_Knot = (int*)malloc(dev_ptrs->LUT_Knot_size);

    int i,j;
    for (i = 0; i < num_tiles; i++) {
        find_knots(local_set_of_64, i, bxf->rdims, bxf->cdims);
        for (j = 0; j < 64; j++) {
            LUT_Knot[64*i + j] = local_set_of_64[j];
        }
    }
    cudaMalloc((void**)&dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->LUT_Knot");
    printf(".");

    cudaMemcpy(dev_ptrs->LUT_Knot, LUT_Knot, dev_ptrs->LUT_Knot_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("cudaMemcpy(): LUT_Knot --> dev_ptrs->LUT_Knot");

    //  cudaBindTexture(0, tex_LUT_Knot, dev_ptrs->LUT_Knot, dev_ptrs->LUT_Knot_size);
    //  cuda_utils_check_error("cudaBindTexture(): dev_ptrs->LUT_Knot");

    free (local_set_of_64);
    free (LUT_Knot);

    GPU_Memory_Bytes += dev_ptrs->LUT_Knot_size;
    // ----------------------------------------------------------

    // --- ALLOCATE CONDENSED dc_dv VECTORS IN GPU GLOBAL -------
    dev_ptrs->cond_x_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_y_size = 64*bxf->num_knots*sizeof(float);
    dev_ptrs->cond_z_size = 64*bxf->num_knots*sizeof(float);

    cudaMalloc((void**)&dev_ptrs->cond_x, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_x");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->cond_y, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_y");
    printf(".");

    cudaMalloc((void**)&dev_ptrs->cond_z, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMalloc(): dev_ptrs->cond_z");
    printf(".");

    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_z");

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
void
bspline_cuda_clean_up_j(Dev_Pointers_Bspline* dev_ptrs)
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
void
bspline_cuda_clean_up_i(Dev_Pointers_Bspline* dev_ptrs)
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
void
bspline_cuda_clean_up_h(Dev_Pointers_Bspline* dev_ptrs)
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



extern "C" int
CUDA_bspline_MI_a_hist (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    Bspline_xform* bxf)
{
    // check to see if we get atomic operations
    // for GPU memory
#ifdef CUDA_NO_SM_12_ATOMIC_INTRINSICS
    printf ("\n******************* FATAL ERROR *******************\n");
    printf ("   Atomic memory operations not supported by GPU!\n");
    printf ("     A GPU of Compute Capability 1.2 or greater\n");
    printf ("     is required to for GPU accelerated MI\n");
    printf ("***************************************************\n\n");
    exit(0);
#endif

    cudaMemset(dev_ptrs->skipped, 0, dev_ptrs->skipped_size);

    // Generate the fixed histogram (48 ms)
    CUDA_bspline_MI_a_hist_fix (dev_ptrs, mi_hist, fixed, moving, bxf);

    // Generate the moving histogram (150 ms)
    CUDA_bspline_MI_a_hist_mov (dev_ptrs, mi_hist, fixed, moving, bxf);

    // Generate the joint histogram (~600 ms)
    return CUDA_bspline_MI_a_hist_jnt (dev_ptrs, mi_hist, fixed, moving, bxf);
}



extern "C" void
CUDA_bspline_MI_a_hist_fix (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    Bspline_xform *bxf)
{
    dim3 dimGrid;
    dim3 dimBlock;
    int num_blocks;

    GPU_Bspline_Data gbd; 
    build_gbd (&gbd, bxf, fixed, moving);

    // Initialize histogram memory on GPU
    cudaMemset(dev_ptrs->f_hist, 0, dev_ptrs->f_hist_size);
    cuda_utils_check_error ("Failed to initialize memory for f_hist");

    num_blocks = build_exec_conf_1tpe (
	&dimGrid,          // OUTPUT: Grid  dimensions
	&dimBlock,         // OUTPUT: Block dimensions
	fixed->npix,       // INPUT: Total # of threads
	32,                // INPUT: Threads per block
	false);            // INPUT: Is threads per block negotiable?

    int smemSize = dimBlock.x * mi_hist->fixed.bins * sizeof(float);

    dev_ptrs->f_hist_seg_size = mi_hist->fixed.bins * num_blocks * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->f_hist_seg, dev_ptrs->f_hist_seg_size);
    cuda_utils_check_error ("Failed to allocate memory for f_hist_seg");
    cudaMemset(dev_ptrs->f_hist_seg, 0, dev_ptrs->f_hist_seg_size);
    cuda_utils_check_error ("Failed to initialize memory for f_hist_seg");


    // Launch kernel with one thread per voxel
    kernel_bspline_MI_a_hist_fix <<<dimGrid, dimBlock, smemSize>>> (
	dev_ptrs->f_hist_seg,       // partial histogram (moving image)
	dev_ptrs->fixed_image,      // moving image voxels
	mi_hist->fixed.offset,      // histogram offset
	1.0f/mi_hist->fixed.delta,  // histogram delta
	mi_hist->fixed.bins,        // # histogram bins
	gbd.vox_per_rgn,            // voxels per region
	gbd.fix_dim,                // fixed  image dimensions
	gbd.mov_dim,                // moving image dimensions
	gbd.rdims,                  //       region dimensions
	gbd.img_origin,             // image origin
	gbd.img_spacing,            // image spacing
	gbd.mov_offset,             // moving image offset
	gbd.mov_spacing,            // moving image pixel spacing
	dev_ptrs->c_lut,            // DEBUG
	dev_ptrs->q_lut,            // DEBUG
	dev_ptrs->coeff);           // DEBUG

    cuda_utils_check_error ("kernel hist_mov");

    int num_sub_hists = num_blocks;

    // Merge sub-histograms
    dim3 dimGrid2 (mi_hist->fixed.bins, 1, 1);
    dim3 dimBlock2 (512, 1, 1);
    smemSize = 512 * sizeof(float);
    
    // this kernel can be ran with any thread-block size
    kernel_bspline_MI_a_hist_fix_merge <<<dimGrid2 , dimBlock2, smemSize>>> (
	dev_ptrs->f_hist,
	dev_ptrs->f_hist_seg,
	num_sub_hists);

    cuda_utils_check_error ("kernel hist_fix_merge");

    cudaMemcpy (mi_hist->f_hist, dev_ptrs->f_hist, dev_ptrs->f_hist_size, 
	cudaMemcpyDeviceToHost);
    cuda_utils_check_error ("Unable to copy fixed histograms from GPU to CPU!\n");

    cudaFree (dev_ptrs->f_hist_seg);
    cuda_utils_check_error ("Error freeing sub-histograms from GPU memory!\n");

}


extern "C" void
CUDA_bspline_MI_a_hist_mov (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    Bspline_xform *bxf)
{
    dim3 dimGrid;
    dim3 dimBlock;
    int num_blocks;

    GPU_Bspline_Data gbd;
    build_gbd (&gbd, bxf, fixed, moving);

    // Initialize histogram memory on GPU
    cudaMemset(dev_ptrs->m_hist, 0, dev_ptrs->m_hist_size);
    cuda_utils_check_error ("Failed to initialize memory for m_hist");
    
    num_blocks = 
	build_exec_conf_1tpe (
	    &dimGrid,          // OUTPUT: Grid  dimensions
	    &dimBlock,         // OUTPUT: Block dimensions
	    fixed->npix,       // INPUT: Total # of threads
	    32,                // INPUT: Threads per block
	    false);            // INPUT: Is threads per block negotiable?

    int smemSize = dimBlock.x * mi_hist->moving.bins * sizeof(float);


    dev_ptrs->m_hist_seg_size = mi_hist->moving.bins * num_blocks * sizeof(float);
    cudaMalloc ((void**)&dev_ptrs->m_hist_seg, dev_ptrs->m_hist_seg_size);
    cuda_utils_check_error ("Failed to allocate memory for m_hist_seg");
    cudaMemset(dev_ptrs->m_hist_seg, 0, dev_ptrs->m_hist_seg_size);
    cuda_utils_check_error ("Failed to initialize memory for m_hist_seg");


    // Launch kernel with one thread per voxel
    kernel_bspline_MI_a_hist_mov <<<dimGrid, dimBlock, smemSize>>> (
	dev_ptrs->m_hist_seg,       // partial histogram (moving image)
	dev_ptrs->moving_image,     // moving image voxels
	mi_hist->moving.offset,     // histogram offset
	1.0f/mi_hist->moving.delta, // histogram delta
	mi_hist->moving.bins,       // # histogram bins
	gbd.vox_per_rgn,            // voxels per region
	gbd.fix_dim,                // fixed  image dimensions
	gbd.mov_dim,                // moving image dimensions
	gbd.rdims,                  //       region dimensions
	gbd.img_origin,             // image origin
	gbd.img_spacing,            // image spacing
	gbd.mov_offset,             // moving image offset
	gbd.mov_spacing,            // moving image pixel spacing
	dev_ptrs->c_lut,            // DEBUG
	dev_ptrs->q_lut,            // DEBUG
	dev_ptrs->coeff);           // DEBUG

    cuda_utils_check_error ("kernel hist_mov");

    int num_sub_hists = num_blocks;


    // Merge sub-histograms
    dim3 dimGrid2 (mi_hist->moving.bins, 1, 1);
    dim3 dimBlock2 (512, 1, 1);
    smemSize = 512 * sizeof(float);
    
    // this kernel can be ran with any thread-block size
    kernel_bspline_MI_a_hist_fix_merge <<<dimGrid2 , dimBlock2, smemSize>>> (
	dev_ptrs->m_hist,
	dev_ptrs->m_hist_seg,
	num_sub_hists);

    cuda_utils_check_error ("kernel hist_mov_merge");

    cudaMemcpy (mi_hist->m_hist, dev_ptrs->m_hist, dev_ptrs->m_hist_size, cudaMemcpyDeviceToHost);
    cuda_utils_check_error ("Unable to copy moving histograms from GPU to CPU!\n");

    cudaFree (dev_ptrs->m_hist_seg);
    cuda_utils_check_error ("Error freeing sub-histograms from GPU memory!\n");

}


extern "C" int
CUDA_bspline_MI_a_hist_jnt (
    Dev_Pointers_Bspline *dev_ptrs,
    BSPLINE_MI_Hist* mi_hist,
    Volume* fixed,
    Volume* moving,
    Bspline_xform *bxf)
{
    GPU_Bspline_Data gbd;
    build_gbd (&gbd, bxf, fixed, moving);

    // Initialize histogram memory on GPU
    cudaMemset(dev_ptrs->j_hist, 0, dev_ptrs->j_hist_size);


    int num_bins = (int)mi_hist->fixed.bins * (int)mi_hist->moving.bins;


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
    // ----------------------

    dev_ptrs->j_hist_seg_size = dev_ptrs->j_hist_size * num_blocks;
    cudaMalloc ((void**)&dev_ptrs->j_hist_seg, dev_ptrs->j_hist_seg_size);
    cudaMemset(dev_ptrs->j_hist_seg, 0, dev_ptrs->j_hist_seg_size);
    cuda_utils_check_error ("Failed to allocate memory for j_hist_seg");
    smemSize = 2 * num_bins * sizeof(float);

    // Launch kernel with one thread per voxel
    kernel_bspline_MI_a_hist_jnt <<<dimGrid1, dimBlock1, smemSize>>> (
            dev_ptrs->skipped,      // # voxels that map outside moving
            dev_ptrs->j_hist_seg,       // partial histogram (moving image)
            dev_ptrs->fixed_image,      // fixed  image voxels
            dev_ptrs->moving_image,     // moving image voxels
            mi_hist->fixed.offset,      // fixed histogram offset
            mi_hist->moving.offset,     // moving histogram offset
            1.0f/mi_hist->fixed.delta,  // fixed histogram delta
            1.0f/mi_hist->moving.delta, // moving histogram delta
            mi_hist->fixed.bins,        // # fixed bins
            mi_hist->moving.bins,       // # moving bins
            gbd.vox_per_rgn,            // voxels per region
            gbd.fix_dim,                // fixed  image dimensions
            gbd.mov_dim,                // moving image dimensions
            gbd.rdims,                  //       region dimensions
            gbd.img_origin,             // image origin
            gbd.img_spacing,            // image spacing
            gbd.mov_offset,             // moving image offset
            gbd.mov_spacing,            // moving image pixel spacing
            gbd.roi_dim,                // region dims
            gbd.roi_offset,             // region offset
            dev_ptrs->c_lut,            // DEBUG
            dev_ptrs->q_lut,            // DEBUG
            dev_ptrs->coeff);           // DEBUG

    cuda_utils_check_error ("kernel hist_jnt");



    // Merge sub-histograms
    threads_per_block = 512;
    dim3 dimGrid2 (num_bins, 1, 1);
    dim3 dimBlock2 (threads_per_block, 1, 1);
    smemSize = 512 * sizeof(float);

    // this kernel can be ran with any thread-block size
    int num_sub_hists = num_blocks;
    kernel_bspline_MI_a_hist_fix_merge <<<dimGrid2 , dimBlock2, smemSize>>> (
    dev_ptrs->j_hist,
    dev_ptrs->j_hist_seg,
    num_sub_hists);

    cuda_utils_check_error ("kernel hist_jnt_merge");

    cudaMemcpy (mi_hist->j_hist, dev_ptrs->j_hist, dev_ptrs->j_hist_size, cudaMemcpyDeviceToHost);
    cuda_utils_check_error ("Unable to copy joint histograms from GPU to CPU!");

    cudaFree (dev_ptrs->j_hist_seg);
    cuda_utils_check_error ("Error freeing sub-histograms from GPU memory!");




    // Use the # of skipped voxels to compute the num_voxels
    // --- INITIALIZE GRID --------------------------------------
    Grid_x = 0;
    Grid_y = 0;
    int num_elems = gbd.fix_dim.x * gbd.fix_dim.y * gbd.fix_dim.z;
    //  int num_blocks = (int)ceil(num_elems / 512.0);
    num_blocks = (num_elems + 511) / 512;
    
    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++)
    {
        if (num_blocks % i == 0) {
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
    //      printf("\nExecuting sum_reduction_kernel() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    dim3 dimBlock(128, 2, 2);
    smemSize = 512 * sizeof(float);
    // ----------------------------------------------------------

    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>>(
            dev_ptrs->skipped,
            dev_ptrs->skipped,
            num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_sum_reduction()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
            dev_ptrs->skipped,
            dev_ptrs->skipped,
            num_elems);
    // ----------------------------------------------------------

    float skipped;
    int num_vox;
    cudaMemcpy(&skipped, dev_ptrs->skipped, sizeof(float), cudaMemcpyDeviceToHost);

    num_vox = (gbd.fix_dim.x * gbd.fix_dim.y * gbd.fix_dim.z) - skipped;
    // -------------------------




    // Now, we back compute bin 0,0 for the joint histogram
    int j = 0;
    for (i = 1; i < mi_hist->fixed.bins * mi_hist->moving.bins; i++) {
        j += mi_hist->j_hist[i];
    }

    mi_hist->j_hist[0] = num_vox - j;

    return num_vox;

}


extern "C" void
CUDA_MI_Grad_a (
    BSPLINE_MI_Hist* mi_hist,
    Bspline_state *bst,
    Bspline_xform *bxf,
    Volume* fixed,
    Volume* moving,
    float num_vox_f,
    Dev_Pointers_Bspline *dev_ptrs)
{
    GPU_Bspline_Data gbd;
    build_gbd (&gbd, bxf, fixed, moving);


    BSPLINE_Score* ssd = &bst->ssd;
    float* host_grad = ssd->grad;
    float score = ssd->score;

    // Initialize histogram memory on GPU
    cudaMemcpy(dev_ptrs->f_hist, mi_hist->f_hist, dev_ptrs->f_hist_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptrs->m_hist, mi_hist->m_hist, dev_ptrs->m_hist_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ptrs->j_hist, mi_hist->j_hist, dev_ptrs->j_hist_size, cudaMemcpyHostToDevice);
       cuda_utils_check_error("CUDA_MI_Grad_a(): Unable to copy histograms to GPU!");

    // Initial dc_dv streams
    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
       cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_x");
    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
       cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_y");
    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);
       cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_z");
    

    // --- INITIALIZE GRID ---
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int threads_per_block = 128;
    int num_threads = fixed->npix;
    int sqrt_num_blocks;
    int num_blocks;
    int found_flag = 0;

    // Search for a valid execution configuration
    // for the required # of blocks.
    for (threads_per_block = 192; threads_per_block > 32; threads_per_block -= 32) {
    num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
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


    int tile_padding = 64 - ((gbd.vox_per_rgn.x * gbd.vox_per_rgn.y * gbd.vox_per_rgn.z) % 64);

    // Launch kernel with one thread per voxel
    kernel_bspline_MI_dc_dv_a <<<dimGrid1, dimBlock1>>> (
        dev_ptrs->dc_dv_x,
        dev_ptrs->dc_dv_y,
        dev_ptrs->dc_dv_z,  
        dev_ptrs->f_hist,
        dev_ptrs->m_hist,
        dev_ptrs->j_hist,
        dev_ptrs->fixed_image,
        dev_ptrs->moving_image,
        mi_hist->fixed.offset,
        mi_hist->moving.offset,
        1.0f/mi_hist->fixed.delta,
        1.0f/mi_hist->moving.delta,
        mi_hist->fixed.bins,
        mi_hist->moving.bins,
        gbd.vox_per_rgn,
        gbd.fix_dim,
        gbd.mov_dim,
        gbd.rdims,
        gbd.img_origin,
        gbd.img_spacing,
        gbd.mov_offset,
        gbd.mov_spacing,
        gbd.roi_dim,
        gbd.roi_offset,
        dev_ptrs->c_lut,
        dev_ptrs->q_lut,
        dev_ptrs->coeff,
        num_vox_f,
        score,
        tile_padding);


    ////////////////////////////////
    // Prepare for the next kernel
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_MI_dc_dv_a()");

    // Clear out the condensed dc_dv streams
    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_x");
    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_y");
    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_z");
    
    // Invoke kernel condense
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    CUDA_bspline_mse_2_condense_64_texfetch (
            dev_ptrs,
            bxf->vox_per_rgn, 
            num_tiles);
    
    // Prepare for the next kernel
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_mse_2_condense_64_texfetch()");

    // Clear out the gradient
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->grad");

    // Invoke kernel reduce
    CUDA_bspline_mse_2_reduce (dev_ptrs, bxf->num_knots);

    // Prepare for the next kernel
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_mse_2_condense()");

    // --- RETREIVE THE GRAD FROM GPU ---------------------------
    cudaMemcpy(host_grad, dev_ptrs->grad, sizeof(float) * bxf->num_coeff, cudaMemcpyDeviceToHost);
    cuda_utils_check_error("Failed to copy dev_ptrs->grad to CPU");
    // ----------------------------------------------------------
}


////////////////////////////////////////////////////////////////////////////////
// STUB: CUDA_deinterleave()
//
// KERNELS INVOKED:
//   kernel_deinterleave()
//
// AUTHOR: James Shackleford
//   DATE: 22 July, 2009
////////////////////////////////////////////////////////////////////////////////
void
CUDA_deinterleave(
    int num_values,
    float* input,
    float* out_x,
    float* out_y,
    float* out_z)
{

    // --- INITIALIZE GRID --------------------------------------
    int i;
    int warps_per_block = 3;    // This cannot be changed.
    int threads_per_block = 32*warps_per_block;
    dim3 dimBlock(threads_per_block, 1, 1);
    int Grid_x = 0;
    int Grid_y = 0;

    int num_blocks = (num_values + threads_per_block - 1) / threads_per_block;


    // *****
    // Search for a valid execution configuration
    // for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);
    
    for (i = sqrt_num_blocks; i < 65535; i++) {
        if (num_blocks % i == 0) {
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
    //      printf("\nExecuting CUDA_deinterleave() with Grid [%i,%i]...\n", Grid_x, Grid_y);
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
        out_z);
    // ----------------------------------------------------------

    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_deinterleave()");
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
extern "C" void
CUDA_pad_64(
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

    for (i = sqrt_num_blocks; i < 65535; i++) {
        if (num_blocks % i == 0) {
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
    //      printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    kernel_pad_64<<<dimGrid, dimBlock>>>(
        *input,
        tmp_output,
        vol_dim,
        tile_dim);
    // ----------------------------------------------------------

    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_pad()");


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
extern "C" void
CUDA_pad(
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

    for (i = sqrt_num_blocks; i < 65535; i++) {
        if (num_blocks % i == 0) {
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
    //      printf("\nExecuting CUDA_row_to_tile_major() with Grid [%i,%i]...\n", Grid_x, Grid_y);
    }

    dim3 dimGrid(Grid_x, Grid_y, 1);
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    kernel_pad<<<dimGrid, dimBlock>>>(
        *input,
        tmp_output,
        vol_dim,
        tile_dim);
    // ----------------------------------------------------------

    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_pad()");


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
extern "C" void
CUDA_bspline_mse_score_dc_dv (
    Dev_Pointers_Bspline* dev_ptrs,
    Bspline_xform* bxf,
    Volume* fixed,
    Volume* moving)
{
    dim3 dimGrid1;
    dim3 dimBlock1;
    GPU_Bspline_Data gbd;   

    build_gbd (&gbd, bxf, fixed, moving);

    build_exec_conf_1tpe (
        &dimGrid1,          // OUTPUT: Grid  dimensions
        &dimBlock1,         // OUTPUT: Block dimensions
        fixed->npix,        // INPUT: Total # of threads
        192,                // INPUT: Threads per block
        true);              // INPUT: Is threads per block negotiable?

#if defined (commentout)
    int smemSize = 12 * sizeof(float) * dimBlock1.x;
#endif

    // --- BEGIN KERNEL EXECUTION ---
    //  cudaEvent_t start, stop;
    //  float time;

    //  cudaEventCreate(&start);
    //  cudaEventCreate(&stop);

    //  cudaEventRecord (start, 0); 

    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_x");

    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_y");

    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_z");

    int tile_padding = 64 - 
    ((gbd.vox_per_rgn.x * gbd.vox_per_rgn.y * gbd.vox_per_rgn.z) % 64);

    /* GCS ??? */
    //if (tile_padding == 64) tile_padding = 0;
    //printf ("tile_padding = %d\n", tile_padding);

    /* JAS 05.27.2010
     * This kernel has been depricated in
     * favor of kernel_bspline_mse_score_dc_dv()
     * and is marked to be moved into bspline_cuda_old.cu
     */
#if defined (commentout)
    bspline_cuda_score_j_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>(
        dev_ptrs->dc_dv_x,      // Addr of dc_dv_x on GPU
        dev_ptrs->dc_dv_y,      // Addr of dc_dv_y on GPU
        dev_ptrs->dc_dv_z,      // Addr of dc_dv_z on GPU
        dev_ptrs->score,        // Addr of score on GPU
        dev_ptrs->coeff,        // Addr of coeff on GPU
        dev_ptrs->fixed_image,  // Addr of fixed_image on GPU
        dev_ptrs->moving_image, // Addr of moving_image on GPU
        dev_ptrs->moving_grad,  // Addr of moving_grad on GPU
        gbd.fix_dim,                // Size of fixed image (vox)
        gbd.img_origin,             // Origin of fixed image (mm)
        gbd.img_spacing,            // Spacing of fixed image (mm)
        gbd.mov_dim,                // Size of moving image (vox)
        gbd.mov_offset,             // Origin of moving image (mm)
        gbd.mov_spacing,            // Spacing of moving image (mm)
        gbd.roi_dim,                // Region of Intrest Dimenions
        gbd.roi_offset,             // Region of Intrest Offset
        gbd.vox_per_rgn,            // Voxels per Region
        gbd.rdims,                  // # of regions in (x,y,z)
        gbd.cdims,                  // # of control points in (x,y,z)
        tile_padding,
        dev_ptrs->skipped);
#endif

    kernel_bspline_mse_score_dc_dv <<<dimGrid1, dimBlock1>>> (
            dev_ptrs->score,
            dev_ptrs->skipped,
            dev_ptrs->dc_dv_x,
            dev_ptrs->dc_dv_y,
            dev_ptrs->dc_dv_z,
            dev_ptrs->fixed_image,
            dev_ptrs->moving_image,
            dev_ptrs->moving_grad,
            gbd.fix_dim,
            gbd.mov_dim,
            gbd.rdims,
            gbd.cdims,
            gbd.vox_per_rgn,
            gbd.img_origin,
            gbd.img_spacing,
            gbd.mov_offset,
            gbd.mov_spacing,
            tile_padding);

    //  cudaEventRecord (stop, 0);  
    //  cudaEventSynchronize (stop);
    //  cudaEventElapsedTime (&time, start, stop);
    //  cudaEventDestroy (start);
    //  cudaEventDestroy (stop);
    //  printf("\n[%f ms] MSE & dc_dv\n", time);
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
void
CUDA_bspline_mse_2_condense_64_texfetch (
    Dev_Pointers_Bspline* dev_ptrs,
    int* vox_per_rgn,
    int num_tiles)
{
    dim3 dimGrid;
    dim3 dimBlock;

    int4 vox_per_region;
    vox_per_region.x = vox_per_rgn[0];
    vox_per_region.y = vox_per_rgn[1];
    vox_per_region.z = vox_per_rgn[2];
    vox_per_region.w = vox_per_region.x * vox_per_region.y * vox_per_region.z;

    int pad = 64 - (vox_per_region.w % 64);

    vox_per_region.w += pad;

    build_exec_conf_1bpe (
        &dimGrid,         // OUTPUT: Grid  dimensions
        &dimBlock,        // OUTPUT: Block dimensions
        num_tiles,        // INPUT: Number of blocks
        64);              // INPUT: Threads per block

    int smemSize = 576*sizeof(float);

    kernel_bspline_mse_2_condense_64_texfetch<<<dimGrid, dimBlock, smemSize>>>(
        dev_ptrs->cond_x,       // Return: condensed dc_dv_x values
        dev_ptrs->cond_y,       // Return: condensed dc_dv_y values
        dev_ptrs->cond_z,       // Return: condensed dc_dv_z values
        dev_ptrs->dc_dv_x,      // Input : dc_dv_x values
        dev_ptrs->dc_dv_y,      // Input : dc_dv_y values
        dev_ptrs->dc_dv_z,      // Input : dc_dv_z values
        dev_ptrs->LUT_Offsets,  // Input : tile offsets
        dev_ptrs->LUT_Knot,     // Input : linear knot indicies
        pad,                    // Input : amount of tile padding
        vox_per_region,         // Input : dims of tiles
        (float)1/6);            // Input : GPU Division is slow
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
void
CUDA_bspline_mse_2_condense_64(
    Dev_Pointers_Bspline* dev_ptrs,
    int* vox_per_rgn,
    int num_tiles)
{
    dim3 dimGrid;
    dim3 dimBlock;

    int4 vox_per_region;
    vox_per_region.x = vox_per_rgn[0];
    vox_per_region.y = vox_per_rgn[1];
    vox_per_region.z = vox_per_rgn[2];
    vox_per_region.w = vox_per_region.x * vox_per_region.y * vox_per_region.z;

    int pad = 64 - (vox_per_region.w % 64);

    vox_per_region.w += pad;

    build_exec_conf_1bpe (
        &dimGrid,         // OUTPUT: Grid  dimensions
        &dimBlock,        // OUTPUT: Block dimensions
        num_tiles,        // INPUT: Number of blocks
        64);              // INPUT: Threads per block

    int smemSize = 384*sizeof(float);

    kernel_bspline_mse_2_condense_64<<<dimGrid, dimBlock, smemSize>>>(
        dev_ptrs->cond_x,       // Return: condensed dc_dv_x values
        dev_ptrs->cond_y,       // Return: condensed dc_dv_y values
        dev_ptrs->cond_z,       // Return: condensed dc_dv_z values
        dev_ptrs->dc_dv_x,      // Input : dc_dv_x values
        dev_ptrs->dc_dv_y,      // Input : dc_dv_y values
        dev_ptrs->dc_dv_z,      // Input : dc_dv_z values
        dev_ptrs->LUT_Offsets,  // Input : tile offsets
        dev_ptrs->LUT_Knot,     // Input : linear knot indicies
        pad,                    // Input : amount of tile padding
        vox_per_region,         // Input : dims of tiles
        (float)1/6);            // Input : GPU Division is slow
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
void
CUDA_bspline_mse_2_condense(
    Dev_Pointers_Bspline* dev_ptrs,
    int* vox_per_rgn,
    int num_tiles)
{
    dim3 dimGrid;
    dim3 dimBlock;

    int4 vox_per_region;
    vox_per_region.x = vox_per_rgn[0];
    vox_per_region.y = vox_per_rgn[1];
    vox_per_region.z = vox_per_rgn[2];
    vox_per_region.w = vox_per_region.x * vox_per_region.y * vox_per_region.z;

    int pad = 32 - (vox_per_region.w % 32);

    build_exec_conf_1bpe (
        &dimGrid,         // OUTPUT: Grid  dimensions
        &dimBlock,        // OUTPUT: Block dimensions
        num_tiles,        // INPUT: Number of blocks
        32);              // INPUT: Threads per block

    int smemSize = 384*sizeof(float);


    kernel_bspline_mse_2_condense<<<dimGrid, dimBlock, smemSize>>>(
        dev_ptrs->cond_x,       // Return: condensed dc_dv_x values
        dev_ptrs->cond_y,       // Return: condensed dc_dv_y values
        dev_ptrs->cond_z,       // Return: condensed dc_dv_z values
        dev_ptrs->dc_dv_x,      // Input : dc_dv_x values
        dev_ptrs->dc_dv_y,      // Input : dc_dv_y values
        dev_ptrs->dc_dv_z,      // Input : dc_dv_z values
        dev_ptrs->LUT_Offsets,  // Input : tile offsets
        dev_ptrs->LUT_Knot,     // Input : linear knot indicies
        pad,                    // Input : amount of tile padding
        vox_per_region,         // Input : dims of tiles
        (float)1/6);            // Input : GPU Division is slow
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
extern "C" void
CUDA_bspline_mse_2_reduce (
    Dev_Pointers_Bspline* dev_ptrs,
    int num_knots)
{
    dim3 dimGrid;
    dim3 dimBlock;

    build_exec_conf_1bpe (
        &dimGrid,         // OUTPUT: Grid  dimensions
        &dimBlock,        // OUTPUT: Block dimensions
        num_knots,        // INPUT: Number of blocks
        64);              // INPUT: Threads per block

    int smemSize = 195*sizeof(float);

    kernel_bspline_mse_2_reduce<<<dimGrid, dimBlock, smemSize>>>(
        dev_ptrs->grad,     // Return: interleaved dc_dp values
        dev_ptrs->cond_x,   // Input : condensed dc_dv_x values
        dev_ptrs->cond_y,   // Input : condensed dc_dv_y values
        dev_ptrs->cond_z);  // Input : condensed dc_dv_z values
}
////////////////////////////////////////////////////////////////////////////////


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
extern "C" void
bspline_cuda_j_stage_1 (
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
    BSPLINE_Parms* parms,
    Dev_Pointers_Bspline* dev_ptrs)
{
#if defined (PROFILE_J)
    cudaEvent_t start, stop;
    float time;
#endif

    // Reset our "voxels fallen outside" counter
    cudaMemset (dev_ptrs->skipped, 0, dev_ptrs->skipped_size);
    cuda_utils_check_error ("cudaMemset(): dev_ptrs->skipped");
    cudaMemset (dev_ptrs->score, 0, dev_ptrs->score_size);
    cuda_utils_check_error ("cudaMemset(): dev_ptrs->score");


#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif


    // Calculate the score and dc_dv
    CUDA_bspline_mse_score_dc_dv (dev_ptrs, bxf, fixed, moving);


#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] score & dc_dv\n", time);
#endif

    // Prepare for the next kernel
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_g_mse_1");

    // Clear out the condensed dc_dv streams
    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_x");
    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_y");
    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_z");


#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif

    // Invoke kernel condense
    int num_tiles = (bxf->cdims[0]-3) * (bxf->cdims[1]-3) * (bxf->cdims[2]-3);
    CUDA_bspline_mse_2_condense_64_texfetch (dev_ptrs,
                                             bxf->vox_per_rgn, 
                                             num_tiles);

#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] Condense\n", time);
#endif

    // Prepare for the next kernel
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_mse_2_condense()");

#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif

    // Clear out the gradient
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->grad");

    // Invoke kernel reduce
    CUDA_bspline_mse_2_reduce (dev_ptrs, bxf->num_knots);

#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] Reduce\n\n", time);
#endif

    // Prepare for the next kernel
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_mse_2_condense()");
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
extern "C" void
bspline_cuda_i_stage_1 (
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
    BSPLINE_Parms* parms,
    Dev_Pointers_Bspline* dev_ptrs)
{
    dim3 dimGrid1;
    dim3 dimBlock1;

    GPU_Bspline_Data gbd;
    build_gbd (&gbd, bxf, fixed, moving);

    build_exec_conf_1tpe (
        &dimGrid1,          // OUTPUT: Grid  dimensions
        &dimBlock1,         // OUTPUT: Block dimensions
        fixed->npix,        // INPUT: Total # of threads
        128,                // INPUT: Threads per block
        false);             // INPUT: Is threads per block negotiable?

    int smemSize = 12 * sizeof(float) * dimBlock1.x;

    // --- BEGIN KERNEL EXECUTION ---
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord (start, 0); 

    cudaMemset(dev_ptrs->dc_dv_x, 0, dev_ptrs->dc_dv_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_x");

    cudaMemset(dev_ptrs->dc_dv_y, 0, dev_ptrs->dc_dv_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_y");

    cudaMemset(dev_ptrs->dc_dv_z, 0, dev_ptrs->dc_dv_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->dc_dv_z");

    cudaMemset(dev_ptrs->skipped, 0, dev_ptrs->skipped_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->skipped");

    int tile_padding = 64 - ((gbd.vox_per_rgn.x * gbd.vox_per_rgn.y * gbd.vox_per_rgn.z) % 64);

    bspline_cuda_score_j_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>(
        dev_ptrs->dc_dv_x,      // Addr of dc_dv_x on GPU
        dev_ptrs->dc_dv_y,      // Addr of dc_dv_y on GPU
        dev_ptrs->dc_dv_z,      // Addr of dc_dv_z on GPU
        dev_ptrs->score,        // Addr of score on GPU
        dev_ptrs->coeff,        // Addr of coeff on GPU
        dev_ptrs->fixed_image,  // Addr of fixed_image on GPU
        dev_ptrs->moving_image, // Addr of moving_image on GPU
        dev_ptrs->moving_grad,  // Addr of moving_grad on GPU
        gbd.fix_dim,            // Size of fixed image (vox)
        gbd.img_origin,         // Origin of fixed image (mm)
        gbd.img_spacing,        // Spacing of fixed image (mm)
        gbd.mov_dim,            // Size of moving image (vox)
        gbd.mov_offset,         // Origin of moving image (mm)
        gbd.mov_spacing,        // Spacing of moving image (mm)
        gbd.roi_dim,            // Region of Intrest Dimenions
        gbd.roi_offset,         // Region of Intrest Offset
        gbd.vox_per_rgn,        // Voxels per Region
        gbd.rdims,              // 
        gbd.cdims,
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
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_g_mse_1");

    // Clear out the condensed dc_dv streams
    cudaMemset(dev_ptrs->cond_x, 0, dev_ptrs->cond_x_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_x");

    cudaMemset(dev_ptrs->cond_y, 0, dev_ptrs->cond_y_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_y");

    cudaMemset(dev_ptrs->cond_z, 0, dev_ptrs->cond_z_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->cond_z");


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
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_mse_2_condense()");

    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);

    // Clear out the gradient
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    cuda_utils_check_error("cudaMemset(): dev_ptrs->grad");

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
    cuda_utils_check_error("[Kernel Panic!] kernel_bspline_mse_2_condense()");

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
extern "C" void
bspline_cuda_j_stage_2 (
    BSPLINE_Parms* parms, 
    Bspline_xform* bxf,
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

#if defined (PROFILE_J)
    cudaEvent_t start, stop;
    float time;
#endif


    dim3 dimGrid;
    dim3 dimBlock;

    int num_elems = volume_dim[0] * volume_dim[1] * volume_dim[2];
    int num_blocks = (num_elems + 511) / 512;

    build_exec_conf_1bpe (
        &dimGrid,         // OUTPUT: Grid  dimensions
        &dimBlock,        // OUTPUT: Block dimensions
        num_blocks,       // INPUT: Number of blocks
        512);             // INPUT: Threads per block

    int smemSize = 512*sizeof(float);


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


#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif

    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>> (
        dev_ptrs->score,
        dev_ptrs->score,
        num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_sum_reduction()");
    // ----------------------------------------------------------

#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] score reduction\n", time);
#endif

    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>> (
        dev_ptrs->score,
        dev_ptrs->score,
        num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_sum_reduction_last_step()");
    // ----------------------------------------------------------


#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif

    // --- RETREIVE THE SCORE FROM GPU --------------------------
    cudaMemcpy(host_score, dev_ptrs->score,  sizeof(float), cudaMemcpyDeviceToHost);
    cuda_utils_check_error("Failed to copy score from GPU to host");
    // ----------------------------------------------------------


#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] score memcpy\n", time);
#endif



#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif



    //  for (i = 1; i < (dev_ptrs->skipped_size / sizeof(int)); i++)
    //      skipped[0] += skipped[i];

    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_kernel<<<dimGrid, dimBlock, smemSize>>> (
        dev_ptrs->skipped,
        dev_ptrs->skipped,
        num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_sum_reduction()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>> (
        dev_ptrs->skipped,
        dev_ptrs->skipped,
        num_elems);
    // ----------------------------------------------------------

    float skipped;
    cudaMemcpy(&skipped, dev_ptrs->skipped, sizeof(float), cudaMemcpyDeviceToHost);

    *num_vox = (volume_dim[0] * volume_dim[1] * volume_dim[2]) - skipped;

    *host_score = *host_score / *num_vox;

#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] skipped reduction\n", time);
#endif



    /////////////////////////////////////////////////////////////
    /////////////////////// CALCULATE ///////////////////////////
    ////////////// GRAD, GRAD NORM *AND* GRAD MEAN //////////////
    /////////////////////////////////////////////////////////////


    num_elems = bxf->num_coeff;
    num_blocks = (num_elems + 511) / 512;

    build_exec_conf_1bpe (
        &dimGrid,         // OUTPUT: Grid  dimensions
        &dimBlock,        // OUTPUT: Block dimensions
        num_blocks,       // INPUT: Number of blocks
        512);             // INPUT: Threads per block


#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif
    
    // --- BEGIN KERNEL EXECUTION -------------------------------
    bspline_cuda_update_grad_kernel<<<dimGrid, dimBlock>>> (
        dev_ptrs->grad,
        *num_vox,
        num_elems);
    // ----------------------------------------------------------


#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] gradient update\n", time);
#endif

    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] bspline_cuda_update_grad_kernel");
    // ----------------------------------------------------------


#if defined (PROFILE_J)
    // Start timing the kernel
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord (start, 0);
#endif

    // --- RETREIVE THE GRAD FROM GPU ---------------------------
    cudaMemcpy(host_grad, dev_ptrs->grad, sizeof(float) * bxf->num_coeff, cudaMemcpyDeviceToHost);
    cuda_utils_check_error("Failed to copy dev_ptrs->grad to CPU");
    // ----------------------------------------------------------


#if defined (PROFILE_J)
    // Stop timing the kernel
    cudaEventRecord (stop, 0);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime (&time, start, stop);
    cudaEventDestroy (start);
    cudaEventDestroy (stop);
    printf("[%f ms] gradient memcpy\n", time);
#endif


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
    cuda_utils_check_error("[Kernel Panic!] bspline_cuda_grad_mean_kernel()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
    dev_ptrs->grad_temp,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_sum_reduction_last_step()");
    // ----------------------------------------------------------


    // --- RETREIVE THE GRAD MEAN FROM GPU ----------------------
    cudaMemcpy(host_grad_mean, dev_ptrs->grad_temp, sizeof(float), cudaMemcpyDeviceToHost);
    cuda_utils_check_error("Failed to copy grad_mean from GPU to host");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize>>>(
    dev_ptrs->grad,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] bspline_cuda_compute_grad_norm_kernel()");
    // ----------------------------------------------------------


    // --- BEGIN KERNEL EXECUTION -------------------------------
    sum_reduction_last_step_kernel<<<dimGrid2, dimBlock2>>>(
    dev_ptrs->grad_temp,
    dev_ptrs->grad_temp,
    num_elems);
    // ----------------------------------------------------------


    // --- PREPARE FOR NEXT KERNEL ------------------------------
    cudaThreadSynchronize();
    cuda_utils_check_error("[Kernel Panic!] kernel_sum_reduction_last_step()");
    // ----------------------------------------------------------


    // --- RETREIVE THE GRAD NORM FROM GPU ----------------------
    cudaMemcpy(host_grad_norm, dev_ptrs->grad_temp, sizeof(float), cudaMemcpyDeviceToHost);
    cuda_utils_check_error("Failed to copy grad_norm from GPU to host");
    // ----------------------------------------------------------
    */
}


////////////////////////////////////////////////////////////////////////////////
// Generates many sub-histograms of the moving image
//
//                 --- Neightborhood of 8 ---
//
// NOTE: The main focus of this kernel is to avoid shared memory
//       bank conflicts.
////////////////////////////////////////////////////////////////////////////////
__global__ void
kernel_bspline_MI_a_hist_fix (
    float* f_hist_seg,  // partial histogram (moving image)
    float* f_img,       // moving image voxels
    float offset,       // histogram offset
    float delta,        // histogram delta
    long bins,          // # histogram bins
    int3 vpr,           // voxels per region
    int3 fdim,          // fixed  image dimensions
    int3 mdim,          // moving image dimensions
    int3 rdim,          //       region dimensions
    float3 img_origin,  // image origin
    float3 img_spacing, // image spacing
    float3 mov_offset,  // moving image offset
    float3 mov_ps,      // moving image pixel spacing
    int* c_lut,         // DEBUG
    float* q_lut,       // DEBUG
    float* coeff)       // DEBUG
{
    // -- Setup Thread Attributes -----------------------------
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;
    // --------------------------------------------------------

    // -- Initialize Shared Memory ----------------------------
    // Amount: 32 * # bins
    extern __shared__ float s_Fixed[];

    for (long i=0; i < bins; i++) {
        s_Fixed[threadIdx.x + i*threadsPerBlock] = 0.0f;
    }
    // --------------------------------------------------------


    __syncthreads();


    // -- Only process threads that map to voxels -------------
    if (thread_idxg > fdim.x * fdim.y * fdim.z) {
        return;
    }
    // --------------------------------------------------------


    // -- Variables used by correspondence --------------------
    // -- (Block verified) ------------------------------------
    int3 r;     // Voxel index (global)
    int4 q;     // Voxel index (local)
    int4 p;     // Tile index


    float3 f;       // Distance from origin (in mm )
    float3 m;       // Voxel Displacement   (in mm )
    float3 n;       // Voxel Displacement   (in vox)
    float3 d;       // Deformation vector

    int fv;     // fixed voxel
    //   ----    ----    ----    ----    ----    ----    ----    
    
    fv = thread_idxg;

    r.z = fv / (fdim.x * fdim.y);
    r.y = (fv - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = fv - r.z * fdim.x * fdim.y - (r.y * fdim.x);
    
    p.x = r.x / vpr.x;
    p.y = r.y / vpr.y;
    p.z = r.z / vpr.z;
    p.w = ((p.z * rdim.y + p.y) * rdim.x) + p.x;

    q.x = r.x - p.x * vpr.x;
    q.y = r.y - p.y * vpr.y;
    q.z = r.z - p.z * vpr.z;
    q.w = ((q.z * vpr.y + q.y) * vpr.x) + q.x;

    f.x = img_origin.x + img_spacing.x * r.x;
    f.y = img_origin.y + img_spacing.y * r.y;
    f.z = img_origin.z + img_spacing.z * r.z;
    // --------------------------------------------------------

#if defined (commentout)
    if (r.x > (roi_offset.x + roi_dim.x) ||
        r.y > (roi_offset.y + roi_dim.y) ||
        r.z > (roi_offset.z + roi_dim.z))
    {
        return;
    }
#endif

    // -- Compute deformation vector --------------------------
    int cidx;
    float P;

    d.x = 0.0f;
    d.y = 0.0f;
    d.z = 0.0f;

    for (int k=0; k < 64; k++) {
        // Texture Version
        P = tex1Dfetch (tex_q_lut, 64*q.w + k);
        cidx = 3 * tex1Dfetch (tex_c_lut, 64*p.w + k);

        d.x += P * tex1Dfetch (tex_coeff, cidx + 0);
        d.y += P * tex1Dfetch (tex_coeff, cidx + 1);
        d.z += P * tex1Dfetch (tex_coeff, cidx + 2);

    
        // Global Memory Version
        //      P = q_lut[64*q.w + k];
        //      cidx = 3 * c_lut[64*p.w + k];
        //
        //      d.x += P * coeff[cidx + 0];
        //      d.y += P * coeff[cidx + 1];
        //      d.z += P * coeff[cidx + 2];
    }
    // --------------------------------------------------------

    float val = 1;

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
        // Voxel doesn't map into the moving image.  
        val = 0;
    }
    // --------------------------------------------------------

    __syncthreads();

    // -- Accumulate Into Segmented Histograms ----------------
    int idx_fbin;
    int f_mem;

    idx_fbin = (int) floorf ((f_img[fv] - offset) * delta);
    f_mem = threadIdx.x + idx_fbin*threadsPerBlock;
    s_Fixed[f_mem] += val;
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
        const int offset   = threadIdx.x * threadsPerBlock;

        for (int i=0, accumPos = startPos; i < threadsPerBlock; i++) {
            sum += s_Fixed[offset + accumPos];
            if (++accumPos == threadsPerBlock) {
                accumPos = 0;
            }
        }

        f_hist_seg[blockIdxInGrid*bins + threadIdx.x] = sum;

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
// Generates many sub-histograms of the moving image
//
//                 --- Neightborhood of 8 ---
//
// NOTE: The main focus of this kernel is to avoid shared memory
//       bank conflicts.
////////////////////////////////////////////////////////////////////////////////
__global__ void
kernel_bspline_MI_a_hist_mov (
    float* m_hist_seg,  // partial histogram (moving image)
    float* m_img,       // moving image voxels
    float offset,       // histogram offset
    float delta,        // histogram delta
    long bins,          // # histogram bins
    int3 vpr,           // voxels per region
    int3 fdim,          // fixed  image dimensions
    int3 mdim,          // moving image dimensions
    int3 rdim,          //       region dimensions
    float3 img_origin,  // image origin
    float3 img_spacing, // image spacing
    float3 mov_offset,  // moving image offset
    float3 mov_ps,      // moving image pixel spacing
    int* c_lut,         // DEBUG
    float* q_lut,       // DEBUG
    float* coeff)       // DEBUG
{
    // -- Setup Thread Attributes -----------------------------
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;
    // --------------------------------------------------------

    // -- Initialize Shared Memory ----------------------------
    // Amount: 32 * # bins
    extern __shared__ float s_Moving[];

    for (long i=0; i < bins; i++) {
        s_Moving[threadIdx.x + i*threadsPerBlock] = 0.0f;
    }
    // --------------------------------------------------------


    __syncthreads();


    // -- Only process threads that map to voxels -------------
    if (thread_idxg > fdim.x * fdim.y * fdim.z) {
        return;
    }
    // --------------------------------------------------------


    // -- Variables used by correspondence --------------------
    // -- (Block verified) ------------------------------------
    int3 r;     // Voxel index (global)
    int4 q;     // Voxel index (local)
    int4 p;     // Tile index


    float3 f;       // Distance from origin (in mm )
    float3 m;       // Voxel Displacement   (in mm )
    float3 n;       // Voxel Displacement   (in vox)
    float3 d;       // Deformation vector

    int3 n_f;       // Voxel Displacement floor

    int fv;     // fixed voxel
    int mvf;        // moving voxel (floor)
    //   ----    ----    ----    ----    ----    ----    ----    
    
    fv = thread_idxg;

    r.z = fv / (fdim.x * fdim.y);
    r.y = (fv - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = fv - r.z * fdim.x * fdim.y - (r.y * fdim.x);
    
    p.x = r.x / vpr.x;
    p.y = r.y / vpr.y;
    p.z = r.z / vpr.z;
    p.w = ((p.z * rdim.y + p.y) * rdim.x) + p.x;

    q.x = r.x - p.x * vpr.x;
    q.y = r.y - p.y * vpr.y;
    q.z = r.z - p.z * vpr.z;
    q.w = ((q.z * vpr.y + q.y) * vpr.x) + q.x;

    f.x = img_origin.x + img_spacing.x * r.x;
    f.y = img_origin.y + img_spacing.y * r.y;
    f.z = img_origin.z + img_spacing.z * r.z;
    // --------------------------------------------------------

#if defined (commentout)
    if (r.x > (roi_offset.x + roi_dim.x) ||
        r.y > (roi_offset.y + roi_dim.y) ||
        r.z > (roi_offset.z + roi_dim.z))
    {
        return;
    }
#endif

    // -- Compute deformation vector --------------------------
    int cidx;
    float P;

    d.x = 0.0f;
    d.y = 0.0f;
    d.z = 0.0f;

    for (int k=0; k < 64; k++) {
        // Texture Version
        P = tex1Dfetch (tex_q_lut, 64*q.w + k);
        cidx = 3 * tex1Dfetch (tex_c_lut, 64*p.w + k);

        d.x += P * tex1Dfetch (tex_coeff, cidx + 0);
        d.y += P * tex1Dfetch (tex_coeff, cidx + 1);
        d.z += P * tex1Dfetch (tex_coeff, cidx + 2);


        // Global Memory Version
        //      P = q_lut[64*q.w + k];
        //      cidx = 3 * c_lut[64*p.w + k];
        //
        //      d.x += P * coeff[cidx + 0];
        //      d.y += P * coeff[cidx + 1];
        //      d.z += P * coeff[cidx + 2];
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
        // Voxel doesn't map into the moving image.
        // Don't add anything to histogram

    } else {

        n_f.x = (int) floorf (n.x);
        n_f.y = (int) floorf (n.y);
        n_f.z = (int) floorf (n.z);
        // --------------------------------------------------------



        // -- Compute tri-linear interpolation weights ------------
        float3 li_1;
        float3 li_2;

        li_2.x = n.x - n_f.x;
        if (n_f.x < 0) {
            n_f.x = 0;
            li_2.x = 0.0f;
        }
        else if (n_f.x >= (mdim.x - 1)) {
            n_f.x = mdim.x - 2;
            li_2.x = 1.0f;
        }
        li_1.x = 1.0f - li_2.x;


        li_2.y = n.y - n_f.y;
        if (n_f.y < 0) {
            n_f.y = 0;
            li_2.y = 0.0f;
        }
        else if (n_f.y >= (mdim.y - 1)) {
            n_f.y = mdim.y - 2;
            li_2.y = 1.0f;
        }
        li_1.y = 1.0f - li_2.y;


        li_2.z = n.z - n_f.z;
        if (n_f.z < 0) {
            n_f.z = 0;
            li_2.z = 0.0f;
        }
        else if (n_f.z >= (mdim.z - 1)) {
            n_f.z = mdim.z - 2;
            li_2.z = 1.0f;
        }
        li_1.z = 1.0f - li_2.z;
        // --------------------------------------------------------


        // -- Compute coordinates of 8 nearest neighbors ----------
        int n1, n2, n3, n4;
        int n5, n6, n7, n8;
    
        mvf = (n_f.z * mdim.y + n_f.y) * mdim.x + n_f.x;

        n1 = mvf;
        n2 = n1 + 1;
        n3 = n1 + mdim.x;
        n4 = n1 + mdim.x + 1;
        n5 = n1 + mdim.x * mdim.y;
        n6 = n1 + mdim.x * mdim.y + 1;
        n7 = n1 + mdim.x * mdim.y + mdim.x;
        n8 = n1 + mdim.x * mdim.y + mdim.x + 1;
        // --------------------------------------------------------


        // -- Compute differential PV slices ----------------------
        float w1, w2, w3, w4;
        float w5, w6, w7, w8;

        w1 = li_1.x * li_1.y * li_1.z;
        w2 = li_2.x * li_1.y * li_1.z;
        w3 = li_1.x * li_2.y * li_1.z;
        w4 = li_2.x * li_2.y * li_1.z;
        w5 = li_1.x * li_1.y * li_2.z;
        w6 = li_2.x * li_1.y * li_2.z;
        w7 = li_1.x * li_2.y * li_2.z;
        w8 = li_2.x * li_2.y * li_2.z;
        // --------------------------------------------------------



        __syncthreads();

        // -- Accumulate Into Segmented Histograms ----------------
        int idx_mbin;
        int m_mem;

        // PV 1
        idx_mbin = (int) floorf ((m_img[n1] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w1;

        // PV 2
        idx_mbin = (int) floorf ((m_img[n2] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w2;

        // PV 3
        idx_mbin = (int) floorf ((m_img[n3] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w3;

        // PV 4
        idx_mbin = (int) floorf ((m_img[n4] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w4;

        // PV 5
        idx_mbin = (int) floorf ((m_img[n5] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w5;

        // PV 6
        idx_mbin = (int) floorf ((m_img[n6] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w6;

        // PV 7
        idx_mbin = (int) floorf ((m_img[n7] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w7;

        // PV 8
        idx_mbin = (int) floorf ((m_img[n8] - offset) * delta);
        m_mem = threadIdx.x + idx_mbin*threadsPerBlock;
        s_Moving[m_mem] += w8;
        // --------------------------------------------------------
    }

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
        const int offset   = threadIdx.x * threadsPerBlock;

        for (int i=0, accumPos = startPos; i < threadsPerBlock; i++) {
            sum += s_Moving[offset + accumPos];
            if (++accumPos == threadsPerBlock) {
                accumPos = 0;
            }
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
__global__ void
kernel_bspline_MI_a_hist_jnt (
    float* skipped, // OUTPUT:   # of skipped voxels
    float* j_hist,      // OUTPUT:  joint histogram
    float* f_img,   // INPUT:  fixed image voxels
    float* m_img,   // INPUT: moving image voxels
    float f_offset, // INPUT:  fixed histogram offset 
    float m_offset, // INPUT: moving histogram offset
    float f_delta,  // INPUT:  fixed histogram delta
    float m_delta,  // INPUT: moving histogram delta
    long f_bins,        // INPUT: #  fixed histogram bins
    long m_bins,    // INPUT: # moving histogram bins
    int3 vpr,       // INPUT: voxels per region
    int3 fdim,      // INPUT:  fixed image dimensions
    int3 mdim,      // INPUT: moving image dimensions
    int3 rdim,      // INPUT: region dimensions
    float3 img_origin,  // INPUT: image origin
    float3 img_spacing, // INPUT: image spacing
    float3 mov_offset,  // INPUT: moving image offset
    float3 mov_ps,  // INPUT: moving image pixel spacing
    int3 roi_dim,   // INPUT: ROI dimensions
    int3 roi_offset,    // INPUT: ROI Offset
    int* c_lut,     // INPUT: coefficient lut
    float* q_lut,   // INPUT: bspline product lut
    float* coeff)   // INPUT: coefficient array
{
    // -- Setup Thread Attributes -----------------------------
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;
    // --------------------------------------------------------

    // -- Initial shared memory for locks ---------------------
    extern __shared__ float shared_mem[]; 

    float* j_locks = (float*)shared_mem;
    int total_smem = f_bins * m_bins;

//    float* sj_hist = (float*)&j_locks[f_bins * m_bins];
//    total_smem += f_bins * m_bins;


    int b = (total_smem + threadsPerBlock - 1) / threadsPerBlock;

    int i;
    for (i = 0; i < b; i++) {
        if ( (thread_idxl + i*threadsPerBlock) < total_smem ) {
            shared_mem[thread_idxl + i*threadsPerBlock] = 0.0f;
        }
    }
    // --------------------------------------------------------


    // -- Only process threads that map to voxels -------------
    if (thread_idxg > fdim.x * fdim.y * fdim.z) {
        return;
    }
    // --------------------------------------------------------


    // -- Variables used by correspondence --------------------
    // -- (Block verified) ------------------------------------
    int3 r;     // Voxel index (global)
    int4 q;     // Voxel index (local)
    int4 p;     // Tile index


    float3 f;       // Distance from origin (in mm )
    float3 m;       // Voxel Displacement   (in mm )
    float3 n;       // Voxel Displacement   (in vox)
    float3 d;       // Deformation vector

    int3 n_f;       // Voxel Displacement floor

    int fv;     // fixed voxel
    int mvf;        // moving voxel (floor)
    //   ----    ----    ----    ----    ----    ----    ----    
    
    fv = thread_idxg;

    r.z = fv / (fdim.x * fdim.y);
    r.y = (fv - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = fv - r.z * fdim.x * fdim.y - (r.y * fdim.x);
    
    p.x = r.x / vpr.x;
    p.y = r.y / vpr.y;
    p.z = r.z / vpr.z;
    p.w = ((p.z * rdim.y + p.y) * rdim.x) + p.x;

    q.x = r.x - p.x * vpr.x;
    q.y = r.y - p.y * vpr.y;
    q.z = r.z - p.z * vpr.z;
    q.w = ((q.z * vpr.y + q.y) * vpr.x) + q.x;

    f.x = img_origin.x + img_spacing.x * r.x;
    f.y = img_origin.y + img_spacing.y * r.y;
    f.z = img_origin.z + img_spacing.z * r.z;
    // --------------------------------------------------------

#if defined (commentout)
    if (r.x > (roi_offset.x + roi_dim.x) ||
        r.y > (roi_offset.y + roi_dim.y) ||
        r.z > (roi_offset.z + roi_dim.z))
    {
        return;
    }
#endif

    // -- Compute deformation vector --------------------------
    int cidx;
    float P;

    d.x = 0.0f;
    d.y = 0.0f;
    d.z = 0.0f;

    for (int k=0; k < 64; k++) {
        // Texture Version
        P = tex1Dfetch (tex_q_lut, 64*q.w + k);
        cidx = 3 * tex1Dfetch (tex_c_lut, 64*p.w + k);

        d.x += P * tex1Dfetch (tex_coeff, cidx + 0);
        d.y += P * tex1Dfetch (tex_coeff, cidx + 1);
        d.z += P * tex1Dfetch (tex_coeff, cidx + 2);


        // Global Memory Version
        //      P = q_lut[64*q.w + k];
        //      cidx = 3 * c_lut[64*p.w + k];
        //
        //      d.x += P * coeff[cidx + 0];
        //      d.y += P * coeff[cidx + 1];
        //      d.z += P * coeff[cidx + 2];
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
        // Voxel doesn't map into the moving image.
        // Don't bin anything and count the miss.

        skipped[thread_idxg]++;
        return;
    }

    n_f.x = (int) floorf (n.x);
    n_f.y = (int) floorf (n.y);
    n_f.z = (int) floorf (n.z);
    // --------------------------------------------------------



    // -- Compute tri-linear interpolation weights ------------
    float3 li_1;
    float3 li_2;

    li_2.x = n.x - n_f.x;
    if (n_f.x < 0) {
        n_f.x = 0;
        li_2.x = 0.0f;
    }
    else if (n_f.x >= (mdim.x - 1)) {
        n_f.x = mdim.x - 2;
        li_2.x = 1.0f;
    }
    li_1.x = 1.0f - li_2.x;


    li_2.y = n.y - n_f.y;
    if (n_f.y < 0) {
        n_f.y = 0;
        li_2.y = 0.0f;
    }
    else if (n_f.y >= (mdim.y - 1)) {
        n_f.y = mdim.y - 2;
        li_2.y = 1.0f;
    }
    li_1.y = 1.0f - li_2.y;


    li_2.z = n.z - n_f.z;
    if (n_f.z < 0) {
        n_f.z = 0;
        li_2.z = 0.0f;
    }
    else if (n_f.z >= (mdim.z - 1)) {
        n_f.z = mdim.z - 2;
        li_2.z = 1.0f;
    }
    li_1.z = 1.0f - li_2.z;
    // --------------------------------------------------------


    // -- Compute coordinates of 8 nearest neighbors ----------
    int n1, n2, n3, n4;
    int n5, n6, n7, n8;

    mvf = (n_f.z * mdim.y + n_f.y) * mdim.x + n_f.x;

    n1 = mvf;
    n2 = n1 + 1;
    n3 = n1 + mdim.x;
    n4 = n1 + mdim.x + 1;
    n5 = n1 + mdim.x * mdim.y;
    n6 = n1 + mdim.x * mdim.y + 1;
    n7 = n1 + mdim.x * mdim.y + mdim.x;
    n8 = n1 + mdim.x * mdim.y + mdim.x + 1;
    // --------------------------------------------------------


    // -- Compute differential PV slices ----------------------
    float w1, w2, w3, w4;
    float w5, w6, w7, w8;

    w1 = li_1.x * li_1.y * li_1.z;
    w2 = li_2.x * li_1.y * li_1.z;
    w3 = li_1.x * li_2.y * li_1.z;
    w4 = li_2.x * li_2.y * li_1.z;
    w5 = li_1.x * li_1.y * li_2.z;
    w6 = li_2.x * li_1.y * li_2.z;
    w7 = li_1.x * li_2.y * li_2.z;
    w8 = li_2.x * li_2.y * li_2.z;
    // --------------------------------------------------------

    __syncthreads();


    // -- Read from histograms and compute dC/dp_j * dp_j/dv --
    bool success;
    int idx_fbin, offset_fbin;
    int idx_mbin;
    int idx_jbin;
    int j_mem;
    long j_bins = f_bins * m_bins;

    long j_stride = blockIdxInGrid * j_bins;

    // Calculate fixed bin offset into joint
    idx_fbin = (int) floorf ((f_img[fv] - f_offset) * f_delta);
    offset_fbin = idx_fbin * m_bins;

    // Add PV w1 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n1] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (idx_jbin != 0) {
        success = false;
        j_mem = j_stride + idx_jbin;
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w1;
               j_hist[j_mem] += w1;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w2 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n2] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (idx_jbin != 0) {
        success = false;
        j_mem = j_stride + idx_jbin;
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w2;
               j_hist[j_mem] += w2;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w3 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n3] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (idx_jbin != 0) {
        success = false;
        j_mem = j_stride + idx_jbin;
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w3;
               j_hist[j_mem] += w3;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w4 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n4] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    success = false;
    j_mem = j_stride + idx_jbin;
    if (idx_jbin != 0) {
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w4;
               j_hist[j_mem] += w4;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w5 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n5] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    success = false;
    j_mem = j_stride + idx_jbin;
    if (idx_jbin != 0) {
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w5;
               j_hist[j_mem] += w5;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w6 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n6] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    success = false;
    j_mem = j_stride + idx_jbin;
    if (idx_jbin != 0) {
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w6;
               j_hist[j_mem] += w6;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w7 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n7] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    success = false;
    j_mem = j_stride + idx_jbin;
    if (idx_jbin != 0) {
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w7;
               j_hist[j_mem] += w7;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

    // Add PV w8 to moving & joint histograms
    idx_mbin = (int) floorf ((m_img[n8] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    success = false;
    j_mem = j_stride + idx_jbin;
    if (idx_jbin != 0) {
        while (!success) {
            if (atomicExch(&j_locks[idx_jbin], 1.0f) == 0.0f) {
               success = true;
//             sj_hist[idx_jbin] += w8;
               j_hist[j_mem] += w8;
               atomicExch(&j_locks[idx_jbin], 0.0f);
            }
            __threadfence();
        }
    }

#if defined (commentout)
    // Almost done...
    // Moving the histogram from shared to global memory
    b = (j_bins + threadsPerBlock - 1) / threadsPerBlock;
    for (i = 0; i < b; i++) {
        if ( (thread_idxl + i*threadsPerBlock) < j_bins ) {
            j_hist[thread_idxl + j_stride] = sj_hist[thread_idxl + i*threadsPerBlock];
        }
    }
#endif
    // --------------------------------------------------------


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
__global__ void
kernel_bspline_MI_a_hist_fix_merge (
    float *f_hist,
    float *f_hist_seg,
    long num_seg_hist)

{
    extern __shared__ float data[];

    float sum = 0.0f;

    // -- Work through all the sub-histograms ------------------------
    for (long i = threadIdx.x; i < num_seg_hist; i += blockDim.x) {
        sum += f_hist_seg[blockIdx.x + i * gridDim.x];
    }

    data[threadIdx.x] = sum;
    // ---------------------------------------------------------------

    __syncthreads();

    // -- Sum all of the thread sums for this bin --------------------
    for (long s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            data[threadIdx.x] += data[threadIdx.x + s];
        }

        __syncthreads();
    }
    // ---------------------------------------------------------------


    // -- Write the final bin value to Global ------------------------
    if (threadIdx.x == 0) {
        f_hist[blockIdx.x] = data[0];
    }
    // ---------------------------------------------------------------

    // Done.
}



////////////////////////////////////////////////////////////////////////////////
// Computes dC/dv for MI using PVI-8 interpolation
//
//                 --- Neightborhood of 8 ---
//
////////////////////////////////////////////////////////////////////////////////
__global__ void
kernel_bspline_MI_dc_dv_a (
    float* dc_dv_x,     // OUTPUT: dC / dv (x-component)
    float* dc_dv_y,     // OUTPUT: dC / dv (y-component)
    float* dc_dv_z,     // OUTPUT: dC / dv (z-component)
    float* f_hist,      // INPUT:  fixed histogram
    float* m_hist,      // INPUT: moving histogram
    float* j_hist,      // INPUT:  joint histogram
    float* f_img,       // INPUT:  fixed image voxels
    float* m_img,       // INPUT: moving image voxels
    float f_offset,     // INPUT:  fixed histogram offset 
    float m_offset,     // INPUT: moving histogram offset
    float f_delta,      // INPUT:  fixed histogram delta
    float m_delta,      // INPUT: moving histogram delta
    long f_bins,        // INPUT: #  fixed histogram bins
    long m_bins,        // INPUT: # moving histogram bins
    int3 vpr,           // INPUT: voxels per region
    int3 fdim,          // INPUT:  fixed image dimensions
    int3 mdim,          // INPUT: moving image dimensions
    int3 rdim,          // INPUT: region dimensions
    float3 img_origin,  // INPUT: image origin
    float3 img_spacing, // INPUT: image spacing
    float3 mov_offset,  // INPUT: moving image offset
    float3 mov_ps,      // INPUT: moving image pixel spacing
    int3 roi_dim,       // INPUT: ROI dimensions
    int3 roi_offset,    // INPUT: ROI Offset
    int* c_lut,         // INPUT: coefficient lut
    float* q_lut,       // INPUT: bspline product lut
    float* coeff,       // INPUT: coefficient array
    float num_vox_f,    // INPUT: # of voxels
    float score,        // INPUT: evaluated MI cost function
    int pad)            // INPUT: Tile padding
{
    // -- Setup Thread Attributes -----------------------------
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;
    // --------------------------------------------------------

    
    // -- Only process threads that map to voxels -------------
    if (thread_idxg > fdim.x * fdim.y * fdim.z) {
        return;
    }
    // --------------------------------------------------------


    // -- Variables used by correspondence --------------------
    // -- (Block verified) ------------------------------------
    int3 r;     // Voxel index (global)
    int4 q;     // Voxel index (local)
    int4 p;     // Tile index


    float3 f;       // Distance from origin (in mm )
    float3 m;       // Voxel Displacement   (in mm )
    float3 n;       // Voxel Displacement   (in vox)
    float3 d;       // Deformation vector

    int3 n_f;       // Voxel Displacement floor

    int fv;     // fixed voxel
    int mvf;        // moving voxel (floor)
    //   ----    ----    ----    ----    ----    ----    ----    
    
    fv = thread_idxg;

    r.z = fv / (fdim.x * fdim.y);
    r.y = (fv - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = fv - r.z * fdim.x * fdim.y - (r.y * fdim.x);
    
    p.x = r.x / vpr.x;
    p.y = r.y / vpr.y;
    p.z = r.z / vpr.z;
    p.w = ((p.z * rdim.y + p.y) * rdim.x) + p.x;

    q.x = r.x - p.x * vpr.x;
    q.y = r.y - p.y * vpr.y;
    q.z = r.z - p.z * vpr.z;
    q.w = ((q.z * vpr.y + q.y) * vpr.x) + q.x;

    f.x = img_origin.x + img_spacing.x * r.x;
    f.y = img_origin.y + img_spacing.y * r.y;
    f.z = img_origin.z + img_spacing.z * r.z;
    // --------------------------------------------------------

    if (r.x > (roi_offset.x + roi_dim.x) ||
        r.y > (roi_offset.y + roi_dim.y) ||
        r.z > (roi_offset.z + roi_dim.z))
    {
        return;
    }

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
        //      P = q_lut[64*q.w + k];
        //      cidx = 3 * c_lut[64*p.w + k];
        //
        //      d.x += P * coeff[cidx + 0];
        //      d.y += P * coeff[cidx + 1];
        //      d.z += P * coeff[cidx + 2];
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
        // if (!rc) continue [in the cpu code]
        return;
    }

    n_f.x = (int) floorf (n.x);
    n_f.y = (int) floorf (n.y);
    n_f.z = (int) floorf (n.z);
    // --------------------------------------------------------



    // -- Compute tri-linear interpolation weights ------------
    float3 li_1;
    float3 li_2;

    li_2.x = n.x - n_f.x;
    if (n_f.x < 0) {
        n_f.x = 0;
        li_2.x = 0.0f;
    }
    else if (n_f.x >= (mdim.x - 1)) {
        n_f.x = mdim.x - 2;
        li_2.x = 1.0f;
    }
    li_1.x = 1.0f - li_2.x;


    li_2.y = n.y - n_f.y;
    if (n_f.y < 0) {
        n_f.y = 0;
        li_2.y = 0.0f;
    }
    else if (n_f.y >= (mdim.y - 1)) {
        n_f.y = mdim.y - 2;
        li_2.y = 1.0f;
    }
    li_1.y = 1.0f - li_2.y;


    li_2.z = n.z - n_f.z;
    if (n_f.z < 0) {
        n_f.z = 0;
        li_2.z = 0.0f;
    }
    else if (n_f.z >= (mdim.z - 1)) {
        n_f.z = mdim.z - 2;
        li_2.z = 1.0f;
    }
    li_1.z = 1.0f - li_2.z;
    // --------------------------------------------------------


    // -- Compute coordinates of 8 nearest neighbors ----------
    int n1, n2, n3, n4;
    int n5, n6, n7, n8;

    mvf = (n_f.z * mdim.y + n_f.y) * mdim.x + n_f.x;

    n1 = mvf;
    n2 = n1 + 1;
    n3 = n1 + mdim.x;
    n4 = n1 + mdim.x + 1;
    n5 = n1 + mdim.x * mdim.y;
    n6 = n1 + mdim.x * mdim.y + 1;
    n7 = n1 + mdim.x * mdim.y + mdim.x;
    n8 = n1 + mdim.x * mdim.y + mdim.x + 1;
    // --------------------------------------------------------


    // -- Compute differential PV slices ----------------------
    float3 dw1, dw2, dw3, dw4;
    float3 dw5, dw6, dw7, dw8;

    dw1.x =  -1.0f * li_1.y * li_1.z;
    dw1.y = li_1.x *  -1.0f * li_1.z;
    dw1.z = li_1.x * li_1.y *  -1.0f;

    dw2.x =  +1.0f * li_1.y * li_1.z;
    dw2.y = li_2.x *  -1.0f * li_1.z;
    dw2.z = li_2.x * li_1.y *  -1.0f;

    dw3.x =  -1.0f * li_2.y * li_1.z;
    dw3.y = li_1.x *  +1.0f * li_1.z;
    dw3.z = li_1.x * li_2.y *  -1.0f;

    dw4.x =  +1.0f * li_2.y * li_1.z;
    dw4.y = li_2.x *  +1.0f * li_1.z;
    dw4.z = li_2.x * li_2.y *  -1.0f;

    dw5.x =  -1.0f * li_1.y * li_2.z;
    dw5.y = li_1.x *  -1.0f * li_2.z;
    dw5.z = li_1.x * li_1.y *  +1.0f;

    dw6.x =  +1.0f * li_1.y * li_2.z;
    dw6.y = li_2.x *  -1.0f * li_2.z;
    dw6.z = li_2.x * li_1.y *  +1.0f;

    dw7.x =  -1.0f * li_2.y * li_2.z;
    dw7.y = li_1.x *  +1.0f * li_2.z;
    dw7.z = li_1.x * li_2.y *  +1.0f;

    dw8.x =  +1.0f * li_2.y * li_2.z;
    dw8.y = li_2.x *  +1.0f * li_2.z;
    dw8.z = li_2.x * li_2.y *  +1.0f;
    // --------------------------------------------------------

    __syncthreads();

    // -- Read from histograms and compute dC/dp_j * dp_j/dv --
    float dS_dP;
    float3 dc_dv;
    int idx_fbin, offset_fbin;
    int idx_mbin;
    int idx_jbin;

    float ht = 0.000001f;

    dc_dv.x = 0.0f;
    dc_dv.y = 0.0f;
    dc_dv.z = 0.0f;


    idx_fbin = (int) floorf ((f_img[fv] - f_offset) * f_delta);
    offset_fbin = idx_fbin * m_bins;

    // PV w1
    idx_mbin = (int) floorf ((m_img[n1] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw1.x * dS_dP;
        dc_dv.y -= dw1.y * dS_dP;
        dc_dv.z -= dw1.z * dS_dP;
    }

    // PV w2
    idx_mbin = (int) floorf ((m_img[n2] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw2.x * dS_dP;
        dc_dv.y -= dw2.y * dS_dP;
        dc_dv.z -= dw2.z * dS_dP;
    }

    // PV w3
    idx_mbin = (int) floorf ((m_img[n3] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw3.x * dS_dP;
        dc_dv.y -= dw3.y * dS_dP;
        dc_dv.z -= dw3.z * dS_dP;
    }

    // PV w4
    idx_mbin = (int) floorf ((m_img[n4] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw4.x * dS_dP;
        dc_dv.y -= dw4.y * dS_dP;
        dc_dv.z -= dw4.z * dS_dP;
    }

    // PV w5
    idx_mbin = (int) floorf ((m_img[n5] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw5.x * dS_dP;
        dc_dv.y -= dw5.y * dS_dP;
        dc_dv.z -= dw5.z * dS_dP;
    }

    // PV w6
    idx_mbin = (int) floorf ((m_img[n6] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw6.x * dS_dP;
        dc_dv.y -= dw6.y * dS_dP;
        dc_dv.z -= dw6.z * dS_dP;
    }

    // PV w7
    idx_mbin = (int) floorf ((m_img[n7] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw7.x * dS_dP;
        dc_dv.y -= dw7.y * dS_dP;
        dc_dv.z -= dw7.z * dS_dP;
    }

    // PV w8
    idx_mbin = (int) floorf ((m_img[n8] - m_offset) * m_delta);
    idx_jbin = offset_fbin + idx_mbin;
    if (j_hist[idx_jbin] > ht && f_hist[idx_fbin] > ht && m_hist[idx_mbin] > ht) {
        dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - score;
        dc_dv.x -= dw8.x * dS_dP;
        dc_dv.y -= dw8.y * dS_dP;
        dc_dv.z -= dw8.z * dS_dP;
    }
    // --------------------------------------------------------


    // -- Convert from voxels to mm ---------------------------
    dc_dv.x = dc_dv.x / mov_ps.x / num_vox_f;
    dc_dv.y = dc_dv.y / mov_ps.y / num_vox_f;
    dc_dv.z = dc_dv.z / mov_ps.z / num_vox_f;
    // --------------------------------------------------------

    __syncthreads();


    // -- Finally, write out dc_dv ----------------------------
    float* dc_dv_element_x;
    float* dc_dv_element_y;
    float* dc_dv_element_z;

    dc_dv_element_x = &dc_dv_x[((vpr.x * vpr.y * vpr.z) + pad) * p.w];
    dc_dv_element_y = &dc_dv_y[((vpr.x * vpr.y * vpr.z) + pad) * p.w];
    dc_dv_element_z = &dc_dv_z[((vpr.x * vpr.y * vpr.z) + pad) * p.w];

    dc_dv_element_x = &dc_dv_element_x[q.w];
    dc_dv_element_y = &dc_dv_element_y[q.w];
    dc_dv_element_z = &dc_dv_element_z[q.w];

    dc_dv_element_x[0] = dc_dv.x;
    dc_dv_element_y[0] = dc_dv.y;
    dc_dv_element_z[0] = dc_dv.z;
    // --------------------------------------------------------


    //NOTE:
    // fv = thread_idxg
    // fi = r.x
    // fj = r.y
    // fk = r.z
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
__global__ void
kernel_deinterleave(
    int num_values,
    float* input,
    float* out_x,
    float* out_y,
    float* out_z)
{
    // Shared memory is allocated on a per block basis.
    // (Allocate (2*96*sizeof(float)) memory when calling the kernel.)
    extern __shared__ float shared_memory[]; 

    float* sdata = (float*)shared_memory;       // float sdata[96];
    float* sdata_x = (float*)&sdata[96];        // float sdata_x[32];
    float* sdata_y = (float*)&sdata_x[32];      // float sdata_y[32];
    float* sdata_z = (float*)&sdata_y[32];      // float sdata_z[32];


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
__global__ void
kernel_row_to_tile_major(
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

    if (threadIdxInGrid >= (vol_dim.x * vol_dim.y * vol_dim.z)) {
        return;
    }

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
    //  output[linear_mem_idx] = (float)threadIdxInGrid;    // Output Check
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
__global__ void
kernel_pad_64(
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

    if (threadIdxInGrid >= (vol_dim.x * vol_dim.y * vol_dim.z)) {
        return;
    }

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
__global__ void
kernel_pad(
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

    if (threadIdxInGrid >= (vol_dim.x * vol_dim.y * vol_dim.z)) {
        return;
    }

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
__global__ void
kernel_bspline_mse_2_condense_64_texfetch (
    float* cond_x,      // Return: condensed dc_dv_x values
    float* cond_y,      // Return: condensed dc_dv_y values
    float* cond_z,      // Return: condensed dc_dv_z values
    float* dc_dv_x,     // Input : dc_dv_x values
    float* dc_dv_y,     // Input : dc_dv_y values
    float* dc_dv_z,     // Input : dc_dv_z values
    int* LUT_Tile_Offsets,  // Input : tile offsets
    int* LUT_Knot,      // Input : linear knot indicies
    int pad,            // Input : amount of tile padding
    int4 tile_dim,      // Input : dims of tiles
    float one_over_six)     // Input : Precomputed since GPU division is slow
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
    float* sBuffer_x = (float*)sdata;           // sBuffer_x[64]
    float* sBuffer_y = (float*)&sBuffer_x[64];      // sBuffer_y[64]
    float* sBuffer_z = (float*)&sBuffer_y[64];      // sBuffer_z[64]
    float* sBuffer_redux_x = (float*)&sBuffer_z[64];    // sBuffer_redux_x[64]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[64];  // sBuffer_redux_y[64]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[64];  // sBuffer_redux_z[64]
    float* sBuffer_redux_x2 = (float*)&sBuffer_redux_z[64]; // sBuffer_redux_x2[64]
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

    tile_pos.w = 0; // Current tile position within [0,63]

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
__global__ void
kernel_bspline_mse_2_condense_64(
    float* cond_x,          // Return: condensed dc_dv_x values
    float* cond_y,          // Return: condensed dc_dv_y values
    float* cond_z,          // Return: condensed dc_dv_z values
    float* dc_dv_x,         // Input : dc_dv_x values
    float* dc_dv_y,         // Input : dc_dv_y values
    float* dc_dv_z,         // Input : dc_dv_z values
    int* LUT_Tile_Offsets,  // Input : tile offsets
    int* LUT_Knot,          // Input : linear knot indices
    int pad,                // Input : amount of tile padding
    int4 tile_dim,          // Input : dims of tiles
    float one_over_six)     // Input : Precomputed since GPU division is slow
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

    int myWarpId_inPair = threadIdxInGrid - 64*blockIdxInGrid;      // From 0 to 63
    // --------------------------------------------------------


    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: 3*threadsPerBlock*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer_x = (float*)sdata;           // sBuffer_x[64]
    float* sBuffer_y = (float*)&sBuffer_x[64];      // sBuffer_y[64]
    float* sBuffer_z = (float*)&sBuffer_y[64];      // sBuffer_z[64]
    float* sBuffer_redux_x = (float*)&sBuffer_z[64];    // sBuffer_redux_x[64]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[64];  // sBuffer_redux_y[64]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[64];  // sBuffer_redux_z[64]
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

    tile_pos.w = 0; // Current tile position within [0,63]

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
    float one_over_six) // Input : Precomputed since GPU division is slow
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

    int myGlobalWarpNumber = threadIdxInGrid / 32;  // Tile #
    int myWarpId = threadIdxInGrid - 32*myGlobalWarpNumber; // 0 to 31
    // --------------------------------------------------------


    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: 3*threadsPerBlock*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer_x = (float*)sdata;           // sBuffer_x[64]
    float* sBuffer_y = (float*)&sBuffer_x[64];      // sBuffer_y[64]
    float* sBuffer_z = (float*)&sBuffer_y[64];      // sBuffer_z[64]
    float* sBuffer_redux_x = (float*)&sBuffer_z[64];    // sBuffer_redux_x[32]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[32];  // sBuffer_redux_y[32]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[32];  // sBuffer_redux_z[32]
    // --------------------------------------------------------


    // Clear Shared Memory!!
    sBuffer_x[myWarpId] = 0; sBuffer_x[myWarpId+32] = 0;
    sBuffer_y[myWarpId] = 0; sBuffer_y[myWarpId+32] = 0;
    sBuffer_z[myWarpId] = 0; sBuffer_z[myWarpId+32] = 0;


    // First, get the offset of where our tile starts in memory.
    //  tileOffset = tex1Dfetch(tex_LUT_Offsets, myGlobalWarpNumber);
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

    tile_pos.w = 0; // Current tile position within [0,63]

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
    float* grad,        // Return: interleaved dc_dp values
    float* cond_x,      // Input : condensed dc_dv_x values
    float* cond_y,      // Input : condensed dc_dv_y values
    float* cond_z)      // Input : condensed dc_dv_z values
{
    // -- Setup Thread Attributes -----------------------------
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    // --------------------------------------------------------

    // -- Setup Shared Memory ---------------------------------
    // -- SIZE: ((3*64)+3)*sizeof(float)
    // --------------------------------------------------------
    extern __shared__ float sdata[]; 
    float* sBuffer = (float*)sdata;             // sBuffer[3]
    float* sBuffer_redux_x = (float*)&sBuffer[3];       // sBuffer_redux_x[64]
    float* sBuffer_redux_y = (float*)&sBuffer_redux_x[64];  // sBuffer_redux_y[64]
    float* sBuffer_redux_z = (float*)&sBuffer_redux_y[64];  // sBuffer_redux_z[64]
    // --------------------------------------------------------

    // Pull down the 64 condensed dc_dv values for the knot this warp pair is working on
    sBuffer_redux_x[threadIdx.x] = cond_x[64*blockIdxInGrid + threadIdx.x];
    sBuffer_redux_y[threadIdx.x] = cond_y[64*blockIdxInGrid + threadIdx.x];
    sBuffer_redux_z[threadIdx.x] = cond_z[64*blockIdxInGrid + threadIdx.x];

    // This thread barrier is very important!
    __syncthreads();
    
    // Perform sum reduction on the 64 condensed dc_dv values
    for(unsigned int s = 32; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
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
    if (threadIdx.x == 0) {
        sBuffer[0] = sBuffer_redux_x[0];
    }
    
    if (threadIdx.x == 1) {
        sBuffer[1] = sBuffer_redux_y[0];
    }

    if (threadIdx.x == 2) {
        sBuffer[2] = sBuffer_redux_z[0];
    }

    // Prevent read before write race condition
    __syncthreads();


    if (threadIdx.x < 3) {
        grad[3*blockIdxInGrid + threadIdx.x] = sBuffer[threadIdx.x];
    }

    // END OF KERNEL 
}


/* JAS 05.27.2010
 * 
 * This kernel was written as an intended replacement for
 * bspline_cuda_score_j_mse_kernel1().  The intended goal
 * was to produce a kernel with neater notation and code
 * structure that also shared the LUT_Bspline_x,y,z textured
 * lookup table that is utilized by the hyper-fast gradient
 * kernel kernel_bspline_mse_2_condense_64_texfetch().
 * 
 * It should be noted that the LUT_Bspline texture differs
 * from the CPU based q_lut in both structure and philosophy.
 * LUT_Bspline is three separate look-up-tables which contain
 * the pre-computed basis function values in each dimension,
 * whereas the q_lut has already pre-multiplied all of these
 * results.  For the GPU, the q-LUT requires in too many memory
 * load operations, even when employing the cacheing mechanisms
 * provided by textures.  The LUT_Bspline textures rely on the GPU
 * to perform these multiplications, thus achieving superior
 * run times.
 *
 * This code was authored with the intention of unifying the
 * design philosophy of the MSE B-spline GPU implementation,
 * which was spurred by my attempts to write the upcoming
 * GPU Gems 4 chapter.
 *
 * The code now also shares more similarities with
 * the CPU code.  So, now if you know one you know the other.
 *
 * This is about 6.5% faster (on my GTX 285) than
 *   bspline_cuda_score_j_mse_kernel1()
 */
__global__ void
kernel_bspline_mse_score_dc_dv (
    float* score,       // OUTPUT
    float* skipped,     // OUTPUT
    float* dc_dv_x,     // OUTPUT
    float* dc_dv_y,     // OUTPUT
    float* dc_dv_z,     // OUTPUT
    float* f_img,       // fixed image voxels
    float* m_img,       // moving image voxels
    float* m_grad,      // moving image gradient
    int3 fdim,          // fixed  image dimensions
    int3 mdim,          // moving image dimensions
    int3 rdim,          //       region dimensions
    int3 cdim,          // # control points in x,y,z
    int3 vpr,           // voxels per region
    float3 img_origin,  // image origin
    float3 img_spacing, // image spacing
    float3 mov_offset,  // moving image offset
    float3 mov_ps,      // moving image pixel spacing
    int pad             // tile padding
)
{
    /* Setup Thread Attributes */
    int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
    int thread_idxl     = (((blockDim.y * threadIdx.z) + threadIdx.y) * blockDim.x) + threadIdx.x;
    int thread_idxg     = (blockIdxInGrid * threadsPerBlock) + thread_idxl;

    /* Only process threads that map to voxels */
    if (thread_idxg > fdim.x * fdim.y * fdim.z) {
        return;
    }

    int4 p;     // Tile index
    int4 q;     // Local Voxel index (within tile)
    float3 f;   // Distance from origin (in mm )

    float3 m;   // Voxel Displacement   (in mm )
    float3 n;   // Voxel Displacement   (in vox)
    int3 n_f;   // Voxel Displacement floor
    int3 n_r;   // Voxel Displacement round
    float3 d;   // Deformation vector
    int fv;     // fixed voxel
    
    fv = thread_idxg;

    setup_indices (&p, &q, &f,
            fv, fdim, vpr, rdim, img_origin, img_spacing);

    int fell_out = find_correspondence (&d, &m, &n,
            f, mov_offset, mov_ps, mdim, cdim, vpr, p, q);

    if (fell_out) {
        skipped[fv]++;
        return;
    }

    float3 li_1, li_2;
    clamp_linear_interpolate_3d (&n, &n_f, &n_r, &li_1, &li_2, mdim);

    float m_val = get_moving_value (n_f, mdim, li_1, li_2);

    float diff = m_val - f_img[fv]; //tex1Dfetch(tex_fixed_image, fv);
    score[fv] = diff * diff;

    write_dc_dv (dc_dv_x, dc_dv_y, dc_dv_z,
            m_grad, diff, n_r, mdim, vpr, pad, p, q);
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
//
// This function is potentially depricated by:
//    kernel_bspline_mse_score_dc_dv()
// and should probably be marked for relocation to bspline_cuda_old.cu
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
    int3   p;             // Index of the tile within the volume (vox)
    int3   q;             // Offset within the tile (measured in voxels)
    int    fv;            // Index of voxel in linear image array
    int    pidx;          // Index into c_lut
    int    qidx;          // Index into q_lut
    int    cidx;          // Index into the coefficient table

    float  P;
    float3 N;             // Multiplier values
    float3 d;             // B-spline deformation vector
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
        diff = m_val - TEX_REF (fixed_image, fv);

        // Accumulate the score.
        score[threadIdxInGrid] = (diff * diff);

        //-----------------------------------------------------------
        // Compute dc_dv for this offset
        //-----------------------------------------------------------
        // Compute spatial gradient using nearest neighbors.
        //      mvr = ((((int)mov_ijk_round.z * mov_dim.y) + (int)mov_ijk_round.y) * mov_dim.x) + (int)mov_ijk_round.x;
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
        //      dc_dv_element_x[0] = diff * TEX_REF (moving_grad, 3 * (int)mvr + 0);
        //      dc_dv_element_y[0] = diff * TEX_REF (moving_grad, 3 * (int)mvr + 1);
        //      dc_dv_element_z[0] = diff * TEX_REF (moving_grad, 3 * (int)mvr + 2);
        }
    }
    }
}




/***********************************************************************
 * bspline_cuda_update_grad_kernel
 *
 * This kernel updates each of the gradient values before the final
 * sum reduction of the gradient stream.
 ***********************************************************************/
__global__ void
bspline_cuda_update_grad_kernel(
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
        //      grad[threadIdxInGrid] = 2.0 * tex1Dfetch(tex_grad, threadIdxInGrid) / num_vox;
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
__global__ void
sum_reduction_kernel(
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
__global__ void
sum_reduction_last_step_kernel(
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
// FUNCTION: bspline_cuda_h_push_coeff_lut()
//
// This function overwries the coefficient LUT to the GPU global
// memory with the new coefficient LUT in preparation for
// the next iteration of score calculation.
////////////////////////////////////////////////////////////////////////////////
void
bspline_cuda_h_push_coeff_lut(Dev_Pointers_Bspline* dev_ptrs, Bspline_xform* bxf)
{
    // Copy the coefficient LUT to the GPU.
    cudaMemcpy(dev_ptrs->coeff, bxf->coeff, dev_ptrs->coeff_size, cudaMemcpyHostToDevice);
    cuda_utils_check_error("[Kernel Panic!] Failed to copy coefficient LUT to GPU");
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_h_clear_score()
//
// This function sets all elements in the score (located on the GPU) to zero
// in preparation for the next iteration of the kernel.
////////////////////////////////////////////////////////////////////////////////
extern "C" void
bspline_cuda_h_clear_score(Dev_Pointers_Bspline* dev_ptrs) 
{
    cudaMemset(dev_ptrs->score, 0, dev_ptrs->score_size);
    cuda_utils_check_error("Failed to clear the score stream on GPU\n");
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_h_clear_grad()
//
// This function sets all elemtns in the gradients (located on the GPU) to
// zero in preparation for the next iteration of the kernel.
////////////////////////////////////////////////////////////////////////////////
extern "C" void
bspline_cuda_h_clear_grad(Dev_Pointers_Bspline* dev_ptrs) 
{
    cudaMemset(dev_ptrs->grad, 0, dev_ptrs->grad_size);
    cuda_utils_check_error("Failed to clear the grad stream on GPU\n");
}
////////////////////////////////////////////////////////////////////////////////





////////////////////////////////////////////////////////////////////////////////
// FUNCTION: CPU_obtain_spline_basis_function()
//
// AUTHOR: James Shackleford
// DATE  : 09.04.2009
////////////////////////////////////////////////////////////////////////////////
float
CPU_obtain_spline_basis_function (
    int t_idx, 
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
        C = one_over_six * (+ 3.0 * i*i*i - 6.0 * i*i           + 4.0);
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


__device__ inline void
clamp_linear_interpolate_3d (
    float3* n,
    int3* n_f,
    int3* n_r,
    float3* li_1,
    float3* li_2,
    int3 mdim
)
{
    /* x-dimension */
    n_f->x = (int) floorf (n->x);
    n_r->x = (int) rintf (n->x);

    li_2->x = n->x - n_f->x;
    if (n_f->x < 0) {
        n_f->x = 0;
        n_r->x = 0;
        li_2->x = 0.0f;
    }
    else if (n_f->x >= (mdim.x - 1)) {
        n_f->x = mdim.x - 2;
        n_r->x = mdim.x - 1;
        li_2->x = 1.0f;
    }
    li_1->x = 1.0f - li_2->x;


    /* y-dimension */
    n_f->y = (int) floorf (n->y);
    n_r->y = (int) rintf (n->y);

    li_2->y = n->y - n_f->y;
    if (n_f->y < 0) {
        n_f->y = 0;
        n_r->y = 0;
        li_2->y = 0.0f;
    }
    else if (n_f->y >= (mdim.y - 1)) {
        n_f->y = mdim.y - 2;
        n_r->y = mdim.y - 1;
        li_2->y = 1.0f;
    }
    li_1->y = 1.0f - li_2->y;


    /* z-dimension */
    n_f->z = (int) floorf (n->z);
    n_r->z = (int) rintf (n->z);

    li_2->z = n->z - n_f->z;
    if (n_f->z < 0) {
        n_f->z = 0;
        n_r->z = 0;
        li_2->z = 0.0f;
    }
    else if (n_f->z >= (mdim.z - 1)) {
        n_f->z = mdim.z - 2;
        n_r->z = mdim.z - 1;
        li_2->z = 1.0f;
    }
    li_1->z = 1.0f - li_2->z;
}


__device__ inline int
find_correspondence (
   float3 *d,
   float3 *m,
   float3 *n,
   float3 f,
   float3 mov_offset,
   float3 mov_ps,
   int3 mdim,
   int3 cdim,
   int3 vpr,
   int4 p,
   int4 q
)
{
    int i, j, k, z, cidx;
    float A,B,C,P;

    d->x = 0.0f;
    d->y = 0.0f;
    d->z = 0.0f;

    z = 0;
    for (k = 0; k < 4; k++) {
    C = tex1Dfetch (tex_LUT_Bspline_x, k * vpr.z + q.z);
        for (j = 0; j < 4; j++) {
        B = tex1Dfetch (tex_LUT_Bspline_x, j * vpr.y + q.y);
            for (i = 0; i < 4; i++) {
                A = tex1Dfetch (tex_LUT_Bspline_x, i * vpr.x + q.x);
                P = A * B * C;

                cidx = 3 * ((p.z + k) * cdim.x * cdim.y 
                            + (p.y + j) * cdim.x + (p.x + i));

                d->x += P * tex1Dfetch (tex_coeff, cidx + 0);
                d->y += P * tex1Dfetch (tex_coeff, cidx + 1);
                d->z += P * tex1Dfetch (tex_coeff, cidx + 2);

                z++;
            }
        }
    }
    // --------------------------------------------------------


    // -- Correspondence --------------------------------------
    m->x = f.x + d->x;  // Displacement in mm
    m->y = f.y + d->y;
    m->z = f.z + d->z;

    // Displacement in voxels
    n->x = (m->x - mov_offset.x) / mov_ps.x;
    n->y = (m->y - mov_offset.y) / mov_ps.y;
    n->z = (m->z - mov_offset.z) / mov_ps.z;

    if (n->x < -0.5 || n->x > mdim.x - 0.5 ||
        n->y < -0.5 || n->y > mdim.y - 0.5 ||
        n->z < -0.5 || n->z > mdim.z - 0.5)
    {
        return 1;
    }
    return 0;
    // --------------------------------------------------------
}

__device__ inline float
get_moving_value (
    int3 n_f,
    int3 mdim,
    float3 li_1,
    float3 li_2
)
{
    // -- Compute coordinates of 8 nearest neighbors ----------
    int mvf;               // moving voxel (floor)
    int n1, n2, n3, n4;    // neighbors
    int n5, n6, n7, n8;
    
    mvf = (n_f.z * mdim.y + n_f.y) * mdim.x + n_f.x;

    n1 = mvf;
    n2 = n1 + 1;
    n3 = n1 + mdim.x;
    n4 = n1 + mdim.x + 1;
    n5 = n1 + mdim.x * mdim.y;
    n6 = n1 + mdim.x * mdim.y + 1;
    n7 = n1 + mdim.x * mdim.y + mdim.x;
    n8 = n1 + mdim.x * mdim.y + mdim.x + 1;
    // --------------------------------------------------------


    // -- Compute Moving Image Intensity ----------------------
    float w1, w2, w3, w4;
    float w5, w6, w7, w8;

    w1 = li_1.x * li_1.y * li_1.z * tex1Dfetch(tex_moving_image, n1);
    w2 = li_2.x * li_1.y * li_1.z * tex1Dfetch(tex_moving_image, n2);
    w3 = li_1.x * li_2.y * li_1.z * tex1Dfetch(tex_moving_image, n3);
    w4 = li_2.x * li_2.y * li_1.z * tex1Dfetch(tex_moving_image, n4);
    w5 = li_1.x * li_1.y * li_2.z * tex1Dfetch(tex_moving_image, n5);
    w6 = li_2.x * li_1.y * li_2.z * tex1Dfetch(tex_moving_image, n6);
    w7 = li_1.x * li_2.y * li_2.z * tex1Dfetch(tex_moving_image, n7);
    w8 = li_2.x * li_2.y * li_2.z * tex1Dfetch(tex_moving_image, n8);

    return w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8;
    // --------------------------------------------------------

}


__device__ inline void
setup_indices (
    int4 *p,
    int4 *q,
    float3 *f,
    int fv,
    int3 fdim,
    int3 vpr,
    int3 rdim,
    float3 img_origin,
    float3 img_spacing
)
{
    /* Setup Global Voxel Indices */
    int3 r;     // Voxel index (global)
    r.z = fv / (fdim.x * fdim.y);
    r.y = (fv - (r.z * fdim.x * fdim.y)) / fdim.x;
    r.x = fv - r.z * fdim.x * fdim.y - (r.y * fdim.x);
    
    /* Setup Tile Indicies */
    p->x = r.x / vpr.x;
    p->y = r.y / vpr.y;
    p->z = r.z / vpr.z;
    p->w = ((p->z * rdim.y + p->y) * rdim.x) + p->x;

    /* Setup Local Voxel Indices */
    q->x = r.x - p->x * vpr.x;
    q->y = r.y - p->y * vpr.y;
    q->z = r.z - p->z * vpr.z;
    q->w = ((q->z * vpr.y + q->y) * vpr.x) + q->x;

    /* Set up fixed image coordinates (mm) */
    f->x = img_origin.x + img_spacing.x * r.x;
    f->y = img_origin.y + img_spacing.y * r.y;
    f->z = img_origin.z + img_spacing.z * r.z;
}


__device__ inline void
write_dc_dv (
    float* dc_dv_x,
    float* dc_dv_y,
    float* dc_dv_z,
    float* m_grad,
    float diff,
    int3 n_r,
    int3 mdim,
    int3 vpr,
    int pad,
    int4 p,
    int4 q
)
{
    float* m_grad_element;
    float* dc_dv_element_x;
    float* dc_dv_element_y;
    float* dc_dv_element_z;

    m_grad_element = &m_grad[3 * n_r.z * mdim.y * mdim.x];
    m_grad_element = &m_grad_element[3 * n_r.y * mdim.x];
    m_grad_element = &m_grad_element[3 * n_r.x];

    dc_dv_element_x = &dc_dv_x[((vpr.x * vpr.y * vpr.z) + pad) * p.w];
    dc_dv_element_y = &dc_dv_y[((vpr.x * vpr.y * vpr.z) + pad) * p.w];
    dc_dv_element_z = &dc_dv_z[((vpr.x * vpr.y * vpr.z) + pad) * p.w];

    dc_dv_element_x = &dc_dv_element_x[q.w];
    dc_dv_element_y = &dc_dv_element_y[q.w];
    dc_dv_element_z = &dc_dv_element_z[q.w];

    dc_dv_element_x[0] = diff * m_grad_element[0];
    dc_dv_element_y[0] = diff * m_grad_element[1];
    dc_dv_element_z[0] = diff * m_grad_element[2];
}
