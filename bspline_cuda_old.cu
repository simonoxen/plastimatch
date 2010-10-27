// texture references
texture<float, 1, cudaReadModeElementType> tex_score;
texture<float, 1> tex_dx;
texture<float, 1> tex_dy;
texture<float, 1> tex_dz;
texture<float, 1> tex_diff;
texture<float, 1> tex_mvr;
texture<float, 1> tex_dc_dv_x;
texture<float, 1> tex_dc_dv_y;
texture<float, 1> tex_dc_dv_z;

// Define global variables.
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


////////////////////////////////////////////
////////////////////////////////////////////
/// SIMPLY UNUSED UTILITY FUNCTIONS    /////
///   STILL VERY GOOD, JUST UN-NEEDED  /////
////////////////////////////////////////////
////////////////////////////////////////////

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



////////////////////////////////////////////
////////////////////////////////////////////
/// DEPRICATED REGISTRATION ALGORITHMS /////
////////////////////////////////////////////
////////////////////////////////////////////

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
// 
// !! REPLACED BY kernel_bspline_mse_score_dc_dv !!
//
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
#if defined (commentout)
extern "C" void
bspline_cuda_i_stage_1 (
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
    Bspline_parms* parms,
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
#endif




/**
 * Calculates the B-spline score and gradient using CUDA implementation H.
 *
 * @param fixed The fixed volume
 * @param moving The moving volume
 * @param moving_grad The spatial gradient of the moving volume
 * @param bxf Pointer to the B-spline Xform
 * @param parms Pointer to the B-spline parameters
 * @param dev_ptrs Pointer the GPU device pointers
 *
 * Author: James A. Shackleford
 */
extern "C" void bspline_cuda_h_stage_1 (Volume* fixed,
				Volume* moving,
				Volume* moving_grad,
				BSPLINE_Xform* bxf,
				BSPLINE_Parms* parms,
				Dev_Pointers_Bspline* dev_ptrs)
{

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
//		printf("\nExecuting bspline_cuda_score_g_mse_kernel1() with Grid [%i,%i]...\n", Grid_x, Grid_y);
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
	CUDA_pad(&dev_ptrs->dc_dv_x,
			fixed->dim,
			bxf->vox_per_rgn);

	CUDA_pad(&dev_ptrs->dc_dv_y,
			fixed->dim,
			bxf->vox_per_rgn);

	CUDA_pad(&dev_ptrs->dc_dv_z,
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
	CUDA_bspline_mse_2_condense(dev_ptrs, bxf->vox_per_rgn, num_tiles);
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

}



////////////////////////////////////////////////////////////////////////////////
// STUB: bspline_cuda_h_stage_2()
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
extern "C" void bspline_cuda_h_stage_2(
	BSPLINE_Parms* parms, 
	BSPLINE_Xform* bxf,
	Volume* fixed,
	int*   vox_per_rgn,
	int*   volume_dim,
	float* host_score,
	float* host_grad,
	float* host_grad_mean,
	float* host_grad_norm,
	Dev_Pointers_Bspline* dev_ptrs)
{

	// --- INITIALIZE GRID --------------------------------------
	int num_elems = volume_dim[0] * volume_dim[1] * volume_dim[2];
	int num_blocks = (int)ceil(num_elems / 512.0);
	dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(128, 2, 2);
	int smemSize = 512 * sizeof(float);
	// ----------------------------------------------------------


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

	*host_score = *host_score / (volume_dim[0] * volume_dim[1] * volume_dim[2]);

	/////////////////////////////////////////////////////////////
	/////////////////////// CALCULATE ///////////////////////////
	////////////// GRAD, GRAD NORM *AND* GRAD MEAN //////////////
	/////////////////////////////////////////////////////////////


	// --- RE-INITIALIZE GRID -----------------------------------
	int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
	num_elems = bxf->num_coeff;
	num_blocks = (int)ceil(num_elems / 512.0);
	dim3 dimGrid2(num_blocks, 1, 1);
	dim3 dimBlock2(128, 2, 2);
	smemSize = 512 * sizeof(float);
	// ----------------------------------------------------------
	

	// --- BEGIN KERNEL EXECUTION -------------------------------
	bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
		dev_ptrs->grad,
		num_vox,
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
}
////////////////////////////////////////////////////////////////////////////////



/***********************************************************************
 * bspline_cuda_score_g_mse_kernel1
 * 
 * This kernel calculates the values for the score and dc_dv streams.
 * It is similar to bspline_cuda_score_f_mse_kernel1, but it computes
 * the c_lut and q_lut values on the fly rather than referencing the
 * lookup tables.
 
 Updated by N. Kandasamy.
 Date: 07 July 2009.
 ***********************************************************************/
__global__ void
bspline_cuda_score_g_mse_kernel1 
(
 float  *dc_dv,
 float  *score,
 float  *coeff,
 float  *fixed_image,
 float  *moving_image,
 float  *moving_grad,
 int3   volume_dim,		// x, y, z dimensions of the volume in voxels
 float3 img_origin,		// Image origin (in mm)
 float3 img_spacing,     // Image spacing (in mm)
 float3 img_offset,		// Offset corresponding to the region of interest
 int3   roi_offset,	    // Position of first vox in ROI (in vox)
 int3   roi_dim,			// Dimension of ROI (in vox)
 int3   vox_per_rgn,	    // Knot spacing (in vox)
 float3 pix_spacing,		// Dimensions of a single voxel (in mm)
 int3   rdims,			// # of regions in (x,y,z)
 int3   cdims)
{
    extern __shared__ float sdata[]; 
	
    int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
    int3   p;				// Index of the tile within the volume (x,y,z)
    int3   q;				// Offset within the tile (measured in voxels)
    int    fv;				// Index of voxel in linear image array
    int    pidx;			// Index into c_lut
    int    qidx;			// Index into q_lut
    int    cidx;			// Index into the coefficient table

    float  P;				
    float3 N;				// Multiplier values
    float3 d;				// B-spline deformation vector
    float  diff;

    float3 distance_from_image_origin;
    float3 displacement_in_mm; 
    float3 displacement_in_vox;
    int3 displacement_in_vox_floor;
    float3 displacement_in_vox_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int    mvf;
    int    mvr;
    float  m_val;
    float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // Allocate memory for the spline coefficients evaluated at indices 0, 1, 2, and 3 in the 
    // X, Y, and Z directions
    float *A = &sdata[12*threadIdxInBlock + 0];
    float *B = &sdata[12*threadIdxInBlock + 4];
    float *C = &sdata[12*threadIdxInBlock + 8];
    float ii, jj, kk;
    float t1, t2, t3; 
    float one_over_six = 1.0/6.0;

    // If the voxel lies outside the volume, do nothing.
    if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
    {	
	// Calculate the x, y, and z coordinate of the voxel within the volume.
	coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
	coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
	coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
	// Calculate the x, y, and z offsets of the tile that contains this voxel.
	p.x = coord_in_volume.x / vox_per_rgn.x;
	p.y = coord_in_volume.y / vox_per_rgn.y;
	p.z = coord_in_volume.z / vox_per_rgn.z;
				
	// Calculate the x, y, and z offsets of the voxel within the tile.
	q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
	q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
	q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
	   coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
	   coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // pidx is the tile index for the tile the current voxel falls within.
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

	    // qidx is the local index of the voxel within the tile
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element = &dc_dv_element[3 * qidx];

	    // Compute the q_lut values that pertain to this offset.
	    ii = ((float)q.x) / vox_per_rgn.x;
	    t3 = ii*ii*ii;
	    t2 = ii*ii;
	    t1 = ii;
	    A[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    A[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    A[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    A[3] = one_over_six * (+ 1.0 * t3);

	    jj = ((float)q.y) / vox_per_rgn.y;
	    t3 = jj*jj*jj;
	    t2 = jj*jj;
	    t1 = jj;
	    B[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    B[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    B[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    B[3] = one_over_six * (+ 1.0 * t3);

	    kk = ((float)q.z) / vox_per_rgn.z;
	    t3 = kk*kk*kk;
	    t2 = kk*kk;
	    t1 = kk;
	    C[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    C[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    C[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    C[3] = one_over_six * (+ 1.0 * t3);

	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    // Compute the B-spline interpolant for the voxel
	    int3 t;
	    for(t.z = 0; t.z < 4; t.z++) {
		for(t.y = 0; t.y < 4; t.y++) {
		    for(t.x = 0; t.x < 4; t.x++) {

			// Calculate the index into the coefficients array.
			cidx = 3 * ((p.z + t.z) * cdims.x * cdims.y + (p.y + t.y) * cdims.x + (p.x + t.x));

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

	    // Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
	    distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
	    distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
	    distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
	    // Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
	    displacement_in_mm.x = distance_from_image_origin.x + d.x;
	    displacement_in_mm.y = distance_from_image_origin.y + d.y;
	    displacement_in_mm.z = distance_from_image_origin.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
	    displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
	    displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

	    // Check if the displaced voxel lies outside the region of interest.
	    if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
		(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
		(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
		// Do nothing.
	    }
	    else {

		//-----------------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------------

		// Clamp and interpolate along the X axis.
		displacement_in_vox_floor.x = (int)(displacement_in_vox.x);
		displacement_in_vox_round.x = rintf(displacement_in_vox.x);	// Single instruction round
		fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
		if(displacement_in_vox_floor.x < 0){
		    displacement_in_vox_floor.x = 0;
		    displacement_in_vox_round.x = 0;
		    fx2 = 0.0;
		}
		else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
		    displacement_in_vox_floor.x = volume_dim.x - 2;
		    displacement_in_vox_round.x = volume_dim.x - 1;
		    fx2 = 1.0;
		}
		fx1 = 1.0 - fx2;

		// Clamp and interpolate along the Y axis.
		displacement_in_vox_floor.y = (int)(displacement_in_vox.y);
		displacement_in_vox_round.y = rintf(displacement_in_vox.y);	// Single instruction round
		fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
		if(displacement_in_vox_floor.y < 0){
		    displacement_in_vox_floor.y = 0;
		    displacement_in_vox_round.y = 0;
		    fy2 = 0.0;
		}
		else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
		    displacement_in_vox_floor.y = volume_dim.y - 2;
		    displacement_in_vox_round.y = volume_dim.y - 1;
		    fy2 = 1.0;
		}
		fy1 = 1.0 - fy2;
				
		// Clamp and intepolate along the Z axis.
		displacement_in_vox_floor.z = (int)(displacement_in_vox.z);
		displacement_in_vox_round.z = rintf(displacement_in_vox.z);	// Single instruction round
		fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
		if(displacement_in_vox_floor.z < 0){
		    displacement_in_vox_floor.z = 0;
		    displacement_in_vox_round.z = 0;
		    fz2 = 0.0;
		}
		else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
		    displacement_in_vox_floor.z = volume_dim.z - 2;
		    displacement_in_vox_round.z = volume_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0 - fz2;
				
		//-----------------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------------

		mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;

		m_x1y1z1 = fx1 * fy1 * fz1 * TEX_REF (moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * TEX_REF (moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * TEX_REF (moving_image, mvf + volume_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * TEX_REF (moving_image, mvf + volume_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * TEX_REF (moving_image, mvf + volume_dim.y * volume_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * TEX_REF (moving_image, mvf + volume_dim.y * volume_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * TEX_REF (moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * TEX_REF (moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);

		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		//-----------------------------------------------------------------
		// Compute intensity difference.
		//-----------------------------------------------------------------

		diff = TEX_REF (fixed_image, fv) - m_val;
				
		//-----------------------------------------------------------------
		// Accumulate the score.
		//-----------------------------------------------------------------

		score[threadIdxInGrid] = (diff * diff);

		//-----------------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------------
				
		// Compute spatial gradient using nearest neighbors.
		mvr = ((((int)displacement_in_vox_round.z * volume_dim.y) + (int)displacement_in_vox_round.y) * volume_dim.x) + (int)displacement_in_vox_round.x;

		float* big_fat_grad;

		big_fat_grad = &moving_grad[3*(int)displacement_in_vox_round.z * volume_dim.y * volume_dim.x];
		big_fat_grad = &big_fat_grad[3*(int)displacement_in_vox_round.y * volume_dim.x];
		big_fat_grad = &big_fat_grad[3*(int)displacement_in_vox_round.x];

                dc_dv_element[0] = diff * big_fat_grad[0];
                dc_dv_element[1] = diff * big_fat_grad[1];
                dc_dv_element[2] = diff * big_fat_grad[2];
	
	    }
	}
    }
}


/******************************************************************
* This function performs the gradient computation. It operates on 
* each control knot is parallel and each control knot accumulates 
* the influence of the 64 tiles on each control knot.

Updated by Naga Kandasamy
Date: 07 July 2009 
*******************************************************************/

__global__ void bspline_cuda_score_g_mse_kernel2 
(
 float *dc_dv,
 float *grad,
 int   num_threads,
 int3  rdims,
 int3  cdims,
 int3  vox_per_rgn)
{
    int3 knotLocation, tileOffset, tileLocation;
    int idx;
    int dc_dv_row;
    float A, B, C;
    int3 q;
    float one_over_six = 1.0/6.0;

    float3 result;
    result.x = 0.0;
    result.y = 0.0;
    result.z = 0.0;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;
	
    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);
	
    // Next, calculate the index of the thread in its thread block, 
    // in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;
	
    // Finally, calculate the index of the thread in the grid, based on 
    // the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // If the thread does not correspond to a control point, do nothing.
    if (threadIdxInGrid >= num_threads) {
	return;
    }

    // Determine the x, y, and z offset of the knot within the grid.
    knotLocation.x = threadIdxInGrid % cdims.x;
    knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
    knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

    // Subtract 1 from each of the knot indices to account for the 
    // differing origin between the knot grid and the tile grid.
    knotLocation.x -= 1;
    knotLocation.y -= 1;
    knotLocation.z -= 1;

    // Iterate through each of the 64 tiles that influence this 
    // control knot.
    for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
	for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
	    for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
		// Using the current x, y, and z offset from the control knot position,
		// calculate the index for one of the tiles that influence this knot.
		tileLocation.x = knotLocation.x + tileOffset.x;
		tileLocation.y = knotLocation.y + tileOffset.y;
		tileLocation.z = knotLocation.z + tileOffset.z;

		// Determine if the tile location is within the volume.
		if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
		   (tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
		   (tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

		    // Calculate linear index for tile.
		    idx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
		    // Calculate the offset into the dc_dv array corresponding to this tile.
		    dc_dv_row = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * idx;
		    int3 t;
		    t.x = abs(tileOffset.x - 1);
		    t.y = abs(tileOffset.y - 1);
		    t.z = abs(tileOffset.z - 1);
						
		    // For all the voxels in this tile, compute the influence on the control knot. We first compute the appropriate 
		    // spline paramterizationfor each voxel, relative to the control knot of interest
						
		    float pre_multiplier;
		    float multiplier_1, multiplier_2, multiplier_3, multiplier_4;
						
		    // Set this parameter to achieve the level of loop 
		    // unrolling desired; could be 1 or 4
		    // An unrolling factor of four appears to be the best 
		    // performer.  
		    int unrolling_factor = 1;
		    // The modified index is an integral multiple of 
		    // the unrolling factor
		    int modified_idx = (vox_per_rgn.x/unrolling_factor)*unrolling_factor; 
		    int lop_off = vox_per_rgn.x - modified_idx;
							
		    // Compute the spline parametization	
		    for(q.z = 0, idx = 0; q.z < vox_per_rgn.z; q.z++) {
			C = obtain_spline_basis_function(one_over_six, t.z, q.z, vox_per_rgn.z);	// Obtain the basis function along the Z direction
			for(q.y = 0; q.y < vox_per_rgn.y; q.y++) {
			    B = obtain_spline_basis_function(one_over_six, t.y, q.y, vox_per_rgn.y); // Obtain the basis function along the Y direction
			    pre_multiplier = B*C;
								
			    // The inner loop is unrolled multiple times as per a specified unrolling factor 
			    for(q.x = 0; q.x < modified_idx; q.x = q.x + unrolling_factor, idx = idx + unrolling_factor) {

				if(unrolling_factor == 1){ // No loop unrolling
				    A = obtain_spline_basis_function(one_over_six, t.x, q.x, vox_per_rgn.x); // Obtain the basis function for voxel in the X direction
				    multiplier_1 = A*pre_multiplier;
										
				    // Accumulate the results
				    // USE GLOBAL MEMORY
//				    result.x += dc_dv[dc_dv_row + 3*idx + 0] * multiplier_1;	
//				    result.y += dc_dv[dc_dv_row + 3*idx + 1] * multiplier_1;	
//				    result.z += dc_dv[dc_dv_row + 3*idx + 2] * multiplier_1;	

				    // USE TEXTURES
				    result.x += TEX_REF (dc_dv, dc_dv_row + 3*idx + 0) * multiplier_1;
				    result.y += TEX_REF (dc_dv, dc_dv_row + 3*idx + 1) * multiplier_1;
				    result.z += TEX_REF (dc_dv, dc_dv_row + 3*idx + 2) * multiplier_1;	


				} // End if unrolling_factor = 1

				if(unrolling_factor == 4){ // The loop is unrolled four times 
				    A = obtain_spline_basis_function(one_over_six, t.x, q.x, vox_per_rgn.x); // Obtain the basis function for Voxel 1 in the X direction
				    multiplier_1 = A * pre_multiplier;
										
				    A = obtain_spline_basis_function(one_over_six, t.x, (q.x + 1), vox_per_rgn.x); // Obtain the basis function for Voxel 2 in the X direction
				    multiplier_2 = A * pre_multiplier;
										
				    A = obtain_spline_basis_function(one_over_six, t.x, (q.x + 2), vox_per_rgn.x); // Obtain the basis function for Voxel 3 in the X direction
				    multiplier_3 = A * pre_multiplier;
										
				    A = obtain_spline_basis_function(one_over_six, t.x, (q.x + 3), vox_per_rgn.x); // Obtain the basis function for Voxel 4 in the X direction
				    multiplier_4 = A * pre_multiplier;
										
				    // Accumulate the results
				    // USE GLOBAL MEMORY
//				    result.x += dc_dv[dc_dv_row + 3*idx + 0] * multiplier_1;
//				    result.y += dc_dv[dc_dv_row + 3*idx + 1] * multiplier_1;
//				    result.z += dc_dv[dc_dv_row + 3*idx + 2] * multiplier_1;
//
//				    result.x += dc_dv[dc_dv_row + 3*(idx + 1) + 0] * multiplier_2;
//				    result.y += dc_dv[dc_dv_row + 3*(idx + 1) + 1] * multiplier_2;
//				    result.z += dc_dv[dc_dv_row + 3*(idx + 1) + 2] * multiplier_2;
//											
//				    result.x += dc_dv[dc_dv_row + 3*(idx + 2) + 0] * multiplier_3;
//				    result.y += dc_dv[dc_dv_row + 3*(idx + 2) + 1] * multiplier_3;
//				    result.z += dc_dv[dc_dv_row + 3*(idx + 2) + 2] * multiplier_3;
//											
//				    result.x += dc_dv[dc_dv_row + 3*(idx + 3) + 0] * multiplier_4;
//				    result.y += dc_dv[dc_dv_row + 3*(idx + 3) + 1] * multiplier_4;
//				    result.z += dc_dv[dc_dv_row + 3*(idx + 3) + 2] * multiplier_4;

				    // USE TEXTURES
				    result.x += TEX_REF (dc_dv, dc_dv_row + 3*idx + 0) * multiplier_1;
				    result.y += TEX_REF (dc_dv, dc_dv_row + 3*idx + 1) * multiplier_1;
				    result.z += TEX_REF (dc_dv, dc_dv_row + 3*idx + 2) * multiplier_1;

				    result.x += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 1) + 0) * multiplier_2;
				    result.y += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 1) + 1) * multiplier_2;
				    result.z += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 1) + 2) * multiplier_2;
											
				    result.x += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 2) + 0) * multiplier_3;
				    result.y += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 2) + 1) * multiplier_3;
				    result.z += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 2) + 2) * multiplier_3;
											
				    result.x += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 3) + 0) * multiplier_4;
				    result.y += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 3) + 1) * multiplier_4;
				    result.z += TEX_REF (dc_dv, dc_dv_row + 3*(idx + 3) + 2) * multiplier_4;

				} // End if unrolling_factor == 4
			    } // End for q.x loop
								
			    // Take care of any lop off voxels that the unrolled loop did not process
			    for(q.x = modified_idx; q.x < (modified_idx + lop_off); q.x++, idx++){
				A = obtain_spline_basis_function(one_over_six, t.x, q.x, vox_per_rgn.x); // Obtain the basis function for voxel in the X direction
				multiplier_1 = A * pre_multiplier;
										
				// Accumulate the results
				// USE GLOBAL MEMORY
//				result.x += dc_dv[dc_dv_row + 3*idx + 0] * multiplier_1;
//				result.y += dc_dv[dc_dv_row + 3*idx + 1] * multiplier_1;
//				result.z += dc_dv[dc_dv_row + 3*idx + 2] * multiplier_1;

				// USE TEXTURES
				result.x += TEX_REF (dc_dv, dc_dv_row + 3*idx + 0) * multiplier_1;
				result.y += TEX_REF (dc_dv, dc_dv_row + 3*idx + 1) * multiplier_1;
				result.z += TEX_REF (dc_dv, dc_dv_row + 3*idx + 2) * multiplier_1;
			    } // End of lop off loop
			} // End for q.y loop
		    } // End q.z loop
		}
	    }
	}
    }


    grad[3*threadIdxInGrid+0] = result.x;
    grad[3*threadIdxInGrid+1] = result.y;
    grad[3*threadIdxInGrid+2] = result.z;
}


/***********************************************************************
 * bspline_cuda_score_g_mse_kernel1_low_mem
 * 
 * This kernel calculates the values for the score and dc_dv streams.
 * It is similar to bspline_cuda_score_f_mse_kernel1, but it computes
 * the c_lut and q_lut values on the fly rather than referencing the
 * lookup tables.  Also, unlike bspline_cuda_score_g_mse_kernel1 above,
 * this version operates on only a portion of the volume at one time
 * in order to reduce the memory requirements on the GPU.
 
 Updated by Naga Kandasamy
 Date: 07 July 2009
 ***********************************************************************/
__global__ void
bspline_cuda_score_g_mse_kernel1_low_mem 
(
 float  *dc_dv,
 float  *score,			
 int    tile_index,		// Linear index of the starting tile
 int    num_tiles,       // Number of tiles to work on per kernel launch
 int3   volume_dim,		// x, y, z dimensions of the volume in voxels
 float3 img_origin,		// Image origin (in mm)
 float3 img_spacing,     // Image spacing (in mm)
 float3 img_offset,		// Offset corresponding to the region of interest
 int3   roi_offset,	    // Position of first vox in ROI (in vox)
 int3   roi_dim,			// Dimension of ROI (in vox)
 int3   vox_per_rgn,	    // Knot spacing (in vox)
 float3 pix_spacing,		// Dimensions of a single voxel (in mm)
 int3   rdims,			// # of regions in (x,y,z)
 int3   cdims)
{
    extern __shared__ float sdata[]; 
	
    int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
    int3   p;				// Offset of the tile within the volume
    int3   q;				// Offset within the tile (measured in voxels)
    int    fv;				// Index of voxel in linear image array
    int    pidx;			// Index into c_lut
    int    qidx;			// Index into q_lut
    int    cidx;			// Index into the coefficient table

    float  P;				
    float3 N;				// Multiplier values
    float3 d;				// B-spline deformation vector
    float  diff;

    float3 distance_from_image_origin;
    float3 displacement_in_mm; 
    float3 displacement_in_vox;
    float3 displacement_in_vox_floor;
    float3 displacement_in_vox_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int    mvf;
    float  mvr;
    float  m_val;
    float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    float *A = &sdata[12*threadIdxInBlock + 0];
    float *B = &sdata[12*threadIdxInBlock + 4];
    float *C = &sdata[12*threadIdxInBlock + 8];
    float ii, jj, kk;
    float t1, t2, t3; 
    float one_over_six = 1.0/6.0;

    // If the voxel lies outside this group of tiles, do nothing.
    if(threadIdxInGrid < (num_tiles * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
    {	
	// Update the tile index to store the index of the tile corresponding to this thread.
	tile_index += (threadIdxInGrid / (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z));

	// Determine the corresponding x, y, and z coordinates of the tile.
	p.x = tile_index % rdims.x;
	p.y = ((tile_index - p.x) / rdims.x) % rdims.y;
	p.z = ((((tile_index - p.x) / rdims.x) - p.y) / rdims.y) % rdims.z;

	// Calculate the x, y and z offsets of the voxel within the tile.
	q.x = threadIdxInGrid % vox_per_rgn.x;
	q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
	q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

	// Calculate the x, y and z offsets of the voxel within the volume.
	coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
	coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
	coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
	   coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
	   coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // Use the offset of the voxel within the region to compute the index into the c_lut.
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

	    // Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element = &dc_dv_element[3 * qidx];
			
	    // Compute the q_lut values that pertain to this offset.
	    ii = ((float)q.x) / vox_per_rgn.x;
	    t3 = ii*ii*ii;
	    t2 = ii*ii;
	    t1 = ii;
	    A[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    A[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    A[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    A[3] = one_over_six * (+ 1.0 * t3);

	    jj = ((float)q.y) / vox_per_rgn.y;
	    t3 = jj*jj*jj;
	    t2 = jj*jj;
	    t1 = jj;
	    B[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    B[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    B[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    B[3] = one_over_six * (+ 1.0 * t3);

	    kk = ((float)q.z) / vox_per_rgn.z;
	    t3 = kk*kk*kk;
	    t2 = kk*kk;
	    t1 = kk;
	    C[0] = one_over_six * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	    C[1] = one_over_six * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	    C[2] = one_over_six * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	    C[3] = one_over_six * (+ 1.0 * t3);

	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    int3 t;
	    for(t.z = 0; t.z < 4; t.z++) {
		for(t.y = 0; t.y < 4; t.y++) {
		    for(t.x = 0; t.x < 4; t.x++) {

			// Calculate the index into the coefficients array.
			cidx = 3 * ((p.z + t.z) * cdims.x * cdims.y + (p.y + t.y) * cdims.x + (p.x + t.x));

			// Fetch the values for P, Ni, Nj, and Nk.
			P   = A[t.x] * B[t.y] * C[t.z];
			N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
			N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
			N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

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

	    // Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
	    distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
	    distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
	    distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
	    // Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
	    displacement_in_mm.x = distance_from_image_origin.x + d.x;
	    displacement_in_mm.y = distance_from_image_origin.y + d.y;
	    displacement_in_mm.z = distance_from_image_origin.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
	    displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
	    displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

	    // Check if the displaced voxel lies outside the region of interest.
	    if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
		(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
		(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
		// Do nothing.
	    }
	    else {

		//-----------------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------------

		// Clamp and interpolate along the X axis.
		displacement_in_vox_floor.x = floor(displacement_in_vox.x);
		displacement_in_vox_round.x = round(displacement_in_vox.x);
		fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
		if(displacement_in_vox_floor.x < 0){
		    displacement_in_vox_floor.x = 0;
		    displacement_in_vox_round.x = 0;
		    fx2 = 0.0;
		}
		else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
		    displacement_in_vox_floor.x = volume_dim.x - 2;
		    displacement_in_vox_round.x = volume_dim.x - 1;
		    fx2 = 1.0;
		}
		fx1 = 1.0 - fx2;

		// Clamp and interpolate along the Y axis.
		displacement_in_vox_floor.y = floor(displacement_in_vox.y);
		displacement_in_vox_round.y = round(displacement_in_vox.y);
		fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
		if(displacement_in_vox_floor.y < 0){
		    displacement_in_vox_floor.y = 0;
		    displacement_in_vox_round.y = 0;
		    fy2 = 0.0;
		}
		else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
		    displacement_in_vox_floor.y = volume_dim.y - 2;
		    displacement_in_vox_round.y = volume_dim.y - 1;
		    fy2 = 1.0;
		}
		fy1 = 1.0 - fy2;
				
		// Clamp and intepolate along the Z axis.
		displacement_in_vox_floor.z = floor(displacement_in_vox.z);
		displacement_in_vox_round.z = round(displacement_in_vox.z);
		fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
		if(displacement_in_vox_floor.z < 0){
		    displacement_in_vox_floor.z = 0;
		    displacement_in_vox_round.z = 0;
		    fz2 = 0.0;
		}
		else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
		    displacement_in_vox_floor.z = volume_dim.z - 2;
		    displacement_in_vox_round.z = volume_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0 - fz2;
				
		//-----------------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------------

		mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
		m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		//-----------------------------------------------------------------
		// Compute intensity difference.
		//-----------------------------------------------------------------

		diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
		//-----------------------------------------------------------------
		// Accumulate the score.
		//-----------------------------------------------------------------

		score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

		//-----------------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------------
				
		// Compute spatial gradient using nearest neighbors.
		mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
		dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
		dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
		dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			
	    }
	}
    }
}
							

/***********************************************************************
 * bspline_cuda_score_f_mse_compute_score
 *
 * This kernel computes only the diff, score, and mvr values.  It stores
 * each in streams to be used by bspline_cuda_score_f_compute_dc_dv. 
 * Separating the score and dc_dv calculations into two separate kernels
 * makes it possible to ensure writes to memory are coalesced.
 ***********************************************************************/
__global__ void 
bspline_cuda_score_f_mse_compute_score 
(
 float  *dc_dv,
 float  *score,
 float  *diffs,
 float  *mvrs,
 int3   volume_dim,		// x, y, z dimensions of the volume in voxels
 float3 img_origin,		// Image origin (in mm)
 float3 img_spacing,     // Image spacing (in mm)
 float3 img_offset,		// Offset corresponding to the region of interest
 int3   roi_offset,	    // Position of first vox in ROI (in vox)
 int3   roi_dim,			// Dimension of ROI (in vox)
 int3   vox_per_rgn,	    // Knot spacing (in vox)
 float3 pix_spacing,		// Dimensions of a single voxel (in mm)
 int3   rdims)			// # of regions in (x,y,z)
{
    int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
    int3   p;				// Index of the tile within the volume (x,y,z)
    int3   q;				// Offset within the tile (measured in voxels)
    int    fv;				// Index of voxel in linear image array
    int    pidx;			// Index into c_lut
    int    qidx;			// Index into q_lut
    int    cidx;			// Index into the coefficient table

    float  P;				
    float3 N;				// Multiplier values
    float3 d;				// B-spline deformation vector
    float  diff;

    float3 distance_from_image_origin;
    float3 displacement_in_mm; 
    float3 displacement_in_vox;
    int3 displacement_in_vox_floor;
    float3 displacement_in_vox_round;
    float  fx1, fx2, fy1, fy2, fz1, fz2;
    int    mvf;
    float  mvr;
    float  m_val;
    float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
    float* dc_dv_element;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    // If the voxel lies outside the volume, do nothing.
    if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
    {	
	// Calculate the x, y, and z coordinate of the voxel within the volume.
	coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
	coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
	coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
	// Calculate the x, y, and z offsets of the tile that contains this voxel.
	p.x = coord_in_volume.x / vox_per_rgn.x;
	p.y = coord_in_volume.y / vox_per_rgn.y;
	p.z = coord_in_volume.z / vox_per_rgn.z;
				
	// Calculate the x, y, and z offsets of the voxel within the tile.
	q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
	q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
	q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

	// If the voxel lies outside of the region of interest, do nothing.
	if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
	   coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
	   coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

	    // Compute the linear index of fixed image voxel.
	    fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

	    //-----------------------------------------------------------------
	    // Calculate the B-Spline deformation vector.
	    //-----------------------------------------------------------------

	    // Use the offset of the voxel within the region to compute the index into the c_lut.
	    pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
	    dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];
	    pidx = pidx * 64;

	    // Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
	    qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
	    dc_dv_element = &dc_dv_element[3 * qidx];
	    qidx = qidx * 64;
			
	    // Compute the deformation vector.
	    d.x = 0.0;
	    d.y = 0.0;
	    d.z = 0.0;

	    for(int k = 0; k < 64; k++)
	    {
		// Calculate the index into the coefficients array.
		cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
		// Fetch the values for P, Ni, Nj, and Nk.
		P   = tex1Dfetch(tex_q_lut, qidx + k); 
		N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
		N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
		N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

		// Update the output (v) values.
		d.x += P * N.x;
		d.y += P * N.y;
		d.z += P * N.z;
	    }
			
	    //-----------------------------------------------------------------
	    // Find correspondence in the moving image.
	    //-----------------------------------------------------------------

	    // Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
	    distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
	    distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
	    distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
	    // Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
	    displacement_in_mm.x = distance_from_image_origin.x + d.x;
	    displacement_in_mm.y = distance_from_image_origin.y + d.y;
	    displacement_in_mm.z = distance_from_image_origin.z + d.z;

	    // Calculate the displacement value in terms of voxels.
	    displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
	    displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
	    displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

	    // Check if the displaced voxel lies outside the region of interest.
	    if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
		(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
		(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
		// Do nothing.
	    }
	    else {

		//-----------------------------------------------------------------
		// Compute interpolation fractions.
		//-----------------------------------------------------------------

		// Clamp and interpolate along the X axis.
		displacement_in_vox_floor.x = (int)(displacement_in_vox.x);
		displacement_in_vox_round.x = rintf(displacement_in_vox.x);	// Single instruction round
		fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
		if(displacement_in_vox_floor.x < 0){
		    displacement_in_vox_floor.x = 0;
		    displacement_in_vox_round.x = 0;
		    fx2 = 0.0;
		}
		else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
		    displacement_in_vox_floor.x = volume_dim.x - 2;
		    displacement_in_vox_round.x = volume_dim.x - 1;
		    fx2 = 1.0;
		}
		fx1 = 1.0 - fx2;

		// Clamp and interpolate along the Y axis.
		displacement_in_vox_floor.y = (int)(displacement_in_vox.y);
		displacement_in_vox_round.y = rintf(displacement_in_vox.y);	// Single instruction round
		fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
		if(displacement_in_vox_floor.y < 0){
		    displacement_in_vox_floor.y = 0;
		    displacement_in_vox_round.y = 0;
		    fy2 = 0.0;
		}
		else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
		    displacement_in_vox_floor.y = volume_dim.y - 2;
		    displacement_in_vox_round.y = volume_dim.y - 1;
		    fy2 = 1.0;
		}
		fy1 = 1.0 - fy2;
				
		// Clamp and intepolate along the Z axis.
		displacement_in_vox_floor.z = (int)(displacement_in_vox.z);
		displacement_in_vox_round.z = rintf(displacement_in_vox.z);	// Single instruction round
		fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
		if(displacement_in_vox_floor.z < 0){
		    displacement_in_vox_floor.z = 0;
		    displacement_in_vox_round.z = 0;
		    fz2 = 0.0;
		}
		else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
		    displacement_in_vox_floor.z = volume_dim.z - 2;
		    displacement_in_vox_round.z = volume_dim.z - 1;
		    fz2 = 1.0;
		}
		fz1 = 1.0 - fz2;
				
		//-----------------------------------------------------------------
		// Compute moving image intensity using linear interpolation.
		//-----------------------------------------------------------------

		mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
		m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
		m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
		m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
		m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
		m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
		m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
		m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
		m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		//-----------------------------------------------------------------
		// Compute intensity difference.
		//-----------------------------------------------------------------

		diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
		//-----------------------------------------------------------------
		// Accumulate the score.
		//-----------------------------------------------------------------

		score[threadIdxInGrid] = (diff * diff);

		diffs[threadIdxInGrid] = diff;

		//-----------------------------------------------------------------
		// Compute dc_dv for this offset
		//-----------------------------------------------------------------
				
		// Compute spatial gradient using nearest neighbors.
		mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
		mvrs[threadIdxInGrid] = (float)mvr;	

		/*
		  dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
		  dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
		  dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
		*/
	    }
	}
    }
}

/***********************************************************************
 * bspline_cuda_score_f_mse_compute_dc_dv
 *
 * This kernel computes only the dc_dv values.
 * Separating the score and dc_dv calculations into two separate kernels
 * makes it possible to ensure writes to memory are coalesced.
 ***********************************************************************/
__global__ void bspline_cuda_score_f_compute_dc_dv(
	float *dc_dv,	
	int3  volume_dim,		// x, y, z dimensions of the volume in voxels
	int3  vox_per_rgn,	    // Knot spacing (in vox)
	int3  roi_offset,	    // Position of first vox in ROI (in vox)
	int3  roi_dim,			// Dimension of ROI (in vox)
	int3  rdims)			// # of regions in (x,y,z)
{	
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	int voxelIdx = threadIdxInGrid / 3;
	int xyzOffset = threadIdxInGrid - (3 * voxelIdx);

	int3  coord_in_volume;	// Coordinate of the voxel in the volume (x,y,z)
	int3  p;				// Index of the tile within the volume (x,y,z)
	int3  q;				// Offset within the tile (measured in voxels)
	int   pidx;				// Index into c_lut
	int   qidx;				// Index into q_lut
	float diff;
	float mvr;
	float *dc_dv_element;

	// If the voxel lies outside the volume, do nothing.
	if(voxelIdx < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the x, y, and z coordinate of the voxel within the volume.
		coord_in_volume.z = voxelIdx / (volume_dim.x * volume_dim.y);
		coord_in_volume.y = (voxelIdx - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		coord_in_volume.x = voxelIdx - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
		// Calculate the x, y, and z offsets of the tile that contains this voxel.
		p.x = coord_in_volume.x / vox_per_rgn.x;
		p.y = coord_in_volume.y / vox_per_rgn.y;
		p.z = coord_in_volume.z / vox_per_rgn.z;
				
		// Calculate the x, y, and z offsets of the voxel within the tile.
		q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
		q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
		q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

		// If the voxel lies outside of the region of interest, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_element = &dc_dv_element[3 * qidx];

			diff = tex1Dfetch(tex_diff, voxelIdx);
			mvr = tex1Dfetch(tex_mvr, voxelIdx);

			dc_dv_element[xyzOffset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + xyzOffset);
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel1_v2
 * 
 * This kernel fills the score and dc_dv streams.  It operates on the
 * entire volume at one time rather than performing the calculations
 * tile by tile.  An equivalent version that operates tile by tile is
 * given below (bspline_cuda_score_f_mse_kernel1_low_mem).  The score
 * stream should have the same number of elements are there are voxels
 * in the volume.
 ***********************************************************************/
__global__ void bspline_cuda_score_f_mse_kernel1_v2 (
	float  *dc_dv_x,
	float  *dc_dv_y,
	float  *dc_dv_z,
	float  *score,			
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	int3   rdims)			// # of regions in (x,y,z)
{
	int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
	int3   p;				// Index of the tile within the volume (x,y,z)
	int3   q;				// Offset within the tile (measured in voxels)
	int    fv;				// Index of voxel in linear image array
	int    pidx;			// Index into c_lut
	int    qidx;			// Index into q_lut
	int    cidx;			// Index into the coefficient table

	float  P;				
	float3 N;				// Multiplier values
	float3 d;				// B-spline deformation vector
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
	int dc_dv_offset;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the x, y, and z coordinate of the voxel within the volume.
		coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
		// Calculate the x, y, and z offsets of the tile that contains this voxel.
		p.x = coord_in_volume.x / vox_per_rgn.x;
		p.y = coord_in_volume.y / vox_per_rgn.y;
		p.z = coord_in_volume.z / vox_per_rgn.z;
				
		// Calculate the x, y, and z offsets of the voxel within the tile.
		q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
		q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
		q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

		// If the voxel lies outside of the region of interest, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_offset = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_offset += qidx;
			qidx = qidx * 64;
			
			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;

				dc_dv_x[dc_dv_offset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv_y[dc_dv_offset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv_z[dc_dv_offset] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);

			}
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel1
 * 
 * This kernel fills the score and dc_dv streams.  It operates on the
 * entire volume at one time rather than performing the calculations
 * tile by tile.  An equivalent version that operates tile by tile is
 * given below (bspline_cuda_score_f_mse_kernel1_low_mem).  The score
 * stream should have the same number of elements are there are voxels
 * in the volume.
 
 Updated by Naga Kandasamy
 Date: 07 July 2009
 ***********************************************************************/
__global__ void bspline_cuda_score_f_mse_kernel1 (
	float  *dc_dv,
	float  *score,	
	int    *gpu_c_lut,
	float  *gpu_q_lut,
	float  *coeff,
	float  *fixed_image,
	float  *moving_image,
	float  *moving_grad,
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	int3   rdims)			// # of regions in (x,y,z)
{
	int3   coord_in_volume; // Coordinate of the voxel in the volume (x,y,z)
	int3   p;				// Index of the tile within the volume (x,y,z)
	int3   q;				// Offset within the tile (measured in voxels)
	int    fv;				// Index of voxel in linear image array
	int    pidx;			// Index into c_lut
	int    qidx;			// Index into q_lut
	int    cidx;			// Index into the coefficient table

	float  P;				
	float3 N;				// Multiplier values
	float3 d;				// B-spline deformation vector
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
	float* dc_dv_element;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{	
		// Calculate the x, y, and z coordinate of the voxel within the volume.
		coord_in_volume.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		coord_in_volume.y = (threadIdxInGrid - (coord_in_volume.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		coord_in_volume.x = threadIdxInGrid - coord_in_volume.z * volume_dim.x * volume_dim.y - (coord_in_volume.y * volume_dim.x);
			
		// Calculate the x, y, and z offsets of the tile that contains this voxel.
		p.x = coord_in_volume.x / vox_per_rgn.x;
		p.y = coord_in_volume.y / vox_per_rgn.y;
		p.z = coord_in_volume.z / vox_per_rgn.z;
				
		// Calculate the x, y, and z offsets of the voxel within the tile.
		q.x = coord_in_volume.x - p.x * vox_per_rgn.x;
		q.y = coord_in_volume.y - p.y * vox_per_rgn.y;
		q.z = coord_in_volume.z - p.z * vox_per_rgn.z;

		// If the voxel lies outside of the region of interest, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			dc_dv_element = &dc_dv_element[3 * qidx]; // dc_dv_element+(3*qidx);

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				// cidx = 3 * gpu_c_lut[pidx + k];

				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				// P   = gpu_q_lut[qidx + k];
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// N.x = coeff[cidx+0];  // x-value
				// N.y = coeff[cidx+1];  // y-value
				// N.z = coeff[cidx+2];  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}

			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				// diff = fixed_image[fv] - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				
				dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
				// dc_dv_element[0] = diff * moving_grad[3 * (int)mvr + 0];
				// dc_dv_element[1] = diff * moving_grad[3 * (int)mvr + 1];
				// dc_dv_element[2] = diff * moving_grad[3 * (int)mvr + 2];	
			}
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_f_mse_kernel1_low_mem
 * 
 * This kernel fills the score and dc_dv streams.  It performs its
 * calculations on a tile by tile basis, and therefore requires the
 * tile index (x, y, and z) as an input.  It uses less memory than 
 * bspline_cuda_score_f_mse_kernel1, but the performance is worse.
 * The score stream need only have the same number of elements as there
 * are voxels in a tile.
 ***********************************************************************/
__global__ void bspline_cuda_score_f_mse_kernel1_low_mem (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
	
	float* dc_dv_element;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInGrid < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			dc_dv_element = &dc_dv[3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx];
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			dc_dv_element = &dc_dv_element[3 * threadIdxInGrid];
			qidx = threadIdxInGrid * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
								
				dc_dv_element[0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv_element[1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv_element[2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

__global__ void bspline_cuda_score_f_mse_kernel2_v2 (
						     float *grad,
						     int   num_threads,
						     int3  rdims,
						     int3  cdims,
						     int3  vox_per_rgn)
{
    // Shared memory is allocated on a per block basis.  Therefore, only allocate 
    // (sizeof(data) * blocksize) memory when calling the kernel.
    extern __shared__ float sdata[]; 

    int3 knotLocation;
    int3 tileOffset;
    int3 tileLocation;
    int pidx;
    int qidx;
    int dc_dv_row;
    int m;	
    float multiplier;

    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    int totalVoxPerRgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

    float *temps = &sdata[15*threadIdxInBlock];
    temps[12] = 0.0;
    temps[13] = 0.0;
    temps[14] = 0.0;

    // If the thread does not correspond to a control point, do nothing.
    if(threadIdxInGrid < num_threads) {	

	// Determine the x, y, and z offset of the knot within the grid.
	knotLocation.x = threadIdxInGrid % cdims.x;
	knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
	knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

	// Subtract 1 from each of the knot indices to account for the differing origin
	// between the knot grid and the tile grid.
	knotLocation.x -= 1;
	knotLocation.y -= 1;
	knotLocation.z -= 1;

	// Iterate through each of the 64 tiles that influence this control knot.
	for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
	    for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
		for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
		    // Using the current x, y, and z offset from the control knot position,
		    // calculate the index for one of the tiles that influence this knot.
		    tileLocation.x = knotLocation.x + tileOffset.x;
		    tileLocation.y = knotLocation.y + tileOffset.y;
		    tileLocation.z = knotLocation.z + tileOffset.z;

		    // Determine if the tile location is within the volume.
		    if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
		       (tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
		       (tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

			// Calculate linear index for tile.
			pidx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
			// Calculate the offset into the dc_dv array corresponding to this tile.
			dc_dv_row = totalVoxPerRgn * pidx;

			// Update pidx to index into the c_lut.
			pidx = 64 * pidx;

			// Find the coefficient index in the c_lut row in order to determine
			// the linear index of the control point with respect to the current tile.
			for(m = 0; m < 64; m++) {
			    if(tex1Dfetch(tex_c_lut, pidx + m) == threadIdxInGrid) {
				break;
			    }
			}

			/*
			  for(qidx = 0; qidx < totalVoxPerRgn; qidx += 1) {
			  multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			  temps[12]  += tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx) * multiplier;
			  temps[13]  += tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx) * multiplier;
			  temps[14]  += tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx) * multiplier;
			  }
			*/

			for(qidx = 0; qidx < totalVoxPerRgn - 4; qidx = qidx + 4) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			    temps[6]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 2) * multiplier;
			    temps[7]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 2) * multiplier;
			    temps[8]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			    temps[9]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 3) * multiplier;
			    temps[10]  = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 3) * multiplier;
			    temps[11]  = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 3) * multiplier;

			    temps[12]  += temps[0] + temps[3] + temps[6] + temps[9];
			    temps[13]  += temps[1] + temps[4] + temps[7] + temps[10];
			    temps[14]  += temps[2] + temps[5] + temps[8] + temps[11];
			}
						
			if(qidx+3 < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			    temps[6]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 2) * multiplier;
			    temps[7]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 2) * multiplier;
			    temps[8]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			    temps[9]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 3) * multiplier;
			    temps[10]  = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 3) * multiplier;
			    temps[11]  = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 3) * multiplier;

			    temps[12]  += temps[0] + temps[3] + temps[6] + temps[9];
			    temps[13]  += temps[1] + temps[4] + temps[7] + temps[10];
			    temps[14]  += temps[2] + temps[5] + temps[8] + temps[11];
			}

			else if(qidx+2 < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			    temps[6]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 2) * multiplier;
			    temps[7]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 2) * multiplier;
			    temps[8]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 2) * multiplier;

			    temps[12]  += temps[0] + temps[3] + temps[6];
			    temps[13]  += temps[1] + temps[4] + temps[7];
			    temps[14]  += temps[2] + temps[5] + temps[8];
			}

			else if(qidx+1 < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[0]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[1]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[2]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			    temps[3]   = tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 1) * multiplier;
			    temps[4]   = tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 1) * multiplier;
			    temps[5]   = tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 1) * multiplier;

			    temps[12]  += temps[0] + temps[3];
			    temps[13]  += temps[1] + temps[4];
			    temps[14]  += temps[2] + temps[5];
			}

			else if(qidx < totalVoxPerRgn) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
			    temps[12]  += tex1Dfetch(tex_dc_dv_x, dc_dv_row + qidx + 0) * multiplier;
			    temps[13]  += tex1Dfetch(tex_dc_dv_y, dc_dv_row + qidx + 0) * multiplier;
			    temps[14]  += tex1Dfetch(tex_dc_dv_z, dc_dv_row + qidx + 0) * multiplier;
			}
		    }
		}
	    }
	}

	grad[3*threadIdxInGrid+0] = temps[12];
	grad[3*threadIdxInGrid+1] = temps[13];
	grad[3*threadIdxInGrid+2] = temps[14];

    }
}

__global__ void bspline_cuda_score_f_mse_kernel2 (
	float *dc_dv,
	float *grad,
	int   num_threads,
	int3  rdims,
	int3  cdims,
	int3  vox_per_rgn)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	int3 knotLocation;
	int3 tileOffset;
	int3 tileLocation;
	int pidx;
	int qidx;
	int dc_dv_row;
	int m;	
	float multiplier;

	/*
	float3 temp0, temp1, temp2, temp3;
	float3 result;
	result.x = 0.0;
	result.y = 0.0;
	result.z = 0.0;
	*/

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	int totalVoxPerRgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

	float *temps = &sdata[15*threadIdxInBlock];
	temps[12] = 0.0;
	temps[13] = 0.0;
	temps[14] = 0.0;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads) {	

		// Determine the x, y, and z offset of the knot within the grid.
		knotLocation.x = threadIdxInGrid % cdims.x;
		knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
		knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

		// Subtract 1 from each of the knot indices to account for the differing origin
		// between the knot grid and the tile grid.
		knotLocation.x -= 1;
		knotLocation.y -= 1;
		knotLocation.z -= 1;

		// Iterate through each of the 64 tiles that influence this control knot.
		for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
			for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
				for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
					// Using the current x, y, and z offset from the control knot position,
					// calculate the index for one of the tiles that influence this knot.
					tileLocation.x = knotLocation.x + tileOffset.x;
					tileLocation.y = knotLocation.y + tileOffset.y;
					tileLocation.z = knotLocation.z + tileOffset.z;

					// Determine if the tile location is within the volume.
					if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
						(tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
						(tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

						// Calculate linear index for tile.
						pidx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
						// Calculate the offset into the dc_dv array corresponding to this tile.
						dc_dv_row = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx;

						// Update pidx to index into the c_lut.
						pidx = 64 * pidx;

						// Find the coefficient index in the c_lut row in order to determine
						// the linear index of the control point with respect to the current tile.
						for(m = 0; m < 64; m++) {
							if(tex1Dfetch(tex_c_lut, pidx + m) == threadIdxInGrid) {
								break;
							}
						}									

						/*
						for(qidx = 0; qidx < totalVoxPerRgn - 4; qidx = qidx + 4) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temp2.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temp2.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temp2.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temp3.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temp3.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temp3.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							result.x += temp0.x + temp1.x + temp2.x + temp3.x;
							result.y += temp0.y + temp1.y + temp2.y + temp3.y;
							result.z += temp0.z + temp1.z + temp2.z + temp3.z;
						}
						
						if(qidx+3 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temp2.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temp2.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temp2.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temp3.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temp3.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temp3.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							result.x += temp0.x + temp1.x + temp2.x + temp3.x;
							result.y += temp0.y + temp1.y + temp2.y + temp3.y;
							result.z += temp0.z + temp1.z + temp2.z + temp3.z;
						}

						else if(qidx+2 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temp2.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temp2.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temp2.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							result.x += temp0.x + temp1.x + temp2.x;
							result.y += temp0.y + temp1.y + temp2.y;
							result.z += temp0.z + temp1.z + temp2.z;
						}

						else if(qidx+1 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temp0.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temp0.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temp0.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temp1.x = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temp1.y = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temp1.z = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							result.x += temp0.x + temp1.x;
							result.y += temp0.y + temp1.y;
							result.z += temp0.z + temp1.z;
						}

						else if(qidx < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;
						}
						*/

						for(qidx = 0; qidx < totalVoxPerRgn - 4; qidx = qidx + 4) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temps[6]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temps[7]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temps[8]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temps[9]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temps[10] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temps[11] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							temps[12] += temps[0] + temps[3] + temps[6] + temps[9];
							temps[13] += temps[1] + temps[4] + temps[7] + temps[10];
							temps[14] += temps[2] + temps[5] + temps[8] + temps[11];
						}
						
						if(qidx+3 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temps[6]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temps[7]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temps[8]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
							temps[9]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
							temps[10] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
							temps[11] = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;

							temps[12] += temps[0] + temps[3] + temps[6] + temps[9];
							temps[13] += temps[1] + temps[4] + temps[7] + temps[10];
							temps[14] += temps[2] + temps[5] + temps[8] + temps[11];
						}

						else if(qidx+2 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
							temps[6]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
							temps[7]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
							temps[8]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

							temps[12] += temps[0] + temps[3] + temps[6];
							temps[13] += temps[1] + temps[4] + temps[7];
							temps[14] += temps[2] + temps[5] + temps[8];
						}

						else if(qidx+1 < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[0]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[1]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[2]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;

							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
							temps[3]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
							temps[4]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
							temps[5]  = tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

							temps[12] += temps[0] + temps[3];
							temps[13] += temps[1] + temps[4];
							temps[14] += temps[2] + temps[5];
						}

						else if(qidx < totalVoxPerRgn) {
							multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+0) + m);
							temps[12] += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 0) * multiplier;
							temps[13] += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 1) * multiplier;
							temps[14] += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+0) + 2) * multiplier;
						}
					}
				}
			}
		}

		/*
		grad[3*threadIdxInGrid+0] = result.x;
		grad[3*threadIdxInGrid+1] = result.y;
		grad[3*threadIdxInGrid+2] = result.z;
		*/

		grad[3*threadIdxInGrid+0] = temps[12];
		grad[3*threadIdxInGrid+1] = temps[13];
		grad[3*threadIdxInGrid+2] = temps[14];

	}
}



/***********************************************************************
 * bspline_cuda_score_f_mse_kernel2
 *
 * This kernel fills up the gradient stream.  Each thread represents one
 * control knot, and therefore one element in the gradient stream.  The
 * kernel determines which tiles influence the given control knot, 
 * iterates through the voxels of each of those tiles, and accumulates
 * the total influence.  It then saves the result to the gradient stream.
 * This implementation offers much better performance than any of the
 * previous versions, which calculate the gradient values on a tile by
 * tile basis.
 ***********************************************************************/
__global__ void 
bspline_cuda_score_f_mse_kernel2_nk 
(
 float *dc_dv,
 float *grad,
 int   num_threads,
 int3  rdims,
 int3  cdims,
 int3  vox_per_rgn)
{
    int3 knotLocation;
    int3 tileOffset;
    int3 tileLocation;
    int pidx;
    int qidx;
    int dc_dv_row;
    int m;	
    float multiplier;

    float3 result;
    result.x = 0.0; 
    result.y = 0.0;
    result.z = 0.0;
	
    // Calculate the index of the thread block in the grid.
    int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

    // Calculate the total number of threads in each thread block.
    int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

    // Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
    int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

    // Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
    int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

    //int totalVoxPerRgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

    // If the thread does not correspond to a control point, do nothing.
    if(threadIdxInGrid < num_threads) {	

	// Determine the x, y, and z offset of the knot within the grid.
	knotLocation.x = threadIdxInGrid % cdims.x;
	knotLocation.y = ((threadIdxInGrid - knotLocation.x) / cdims.x) % cdims.y;
	knotLocation.z = ((((threadIdxInGrid - knotLocation.x) / cdims.x) - knotLocation.y) / cdims.y) % cdims.z;

	// Subtract 1 from each of the knot indices to account for the differing origin
	// between the knot grid and the tile grid.
	knotLocation.x -= 1;
	knotLocation.y -= 1;
	knotLocation.z -= 1;

	// Iterate through each of the 64 tiles that influence this control knot.
	for(tileOffset.z = -2; tileOffset.z < 2; tileOffset.z++) {
	    for(tileOffset.y = -2; tileOffset.y < 2; tileOffset.y++) {
		for(tileOffset.x = -2; tileOffset.x < 2; tileOffset.x++) {
						
		    // Using the current x, y, and z offset from the control knot position,
		    // calculate the index for one of the tiles that influence this knot.
		    tileLocation.x = knotLocation.x + tileOffset.x;
		    tileLocation.y = knotLocation.y + tileOffset.y;
		    tileLocation.z = knotLocation.z + tileOffset.z;

		    // Determine if the tile location is within the volume.
		    if((tileLocation.x >= 0 && tileLocation.x < rdims.x) &&
		       (tileLocation.y >= 0 && tileLocation.y < rdims.y) &&
		       (tileLocation.z >= 0 && tileLocation.z < rdims.z)) {

			// Calculate linear index for tile.
			pidx = ((tileLocation.z * rdims.y + tileLocation.y) * rdims.x) + tileLocation.x;	
						
			// Calculate the offset into the dc_dv array corresponding to this tile.
			dc_dv_row = 3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z * pidx;

			// Update pidx to index into the c_lut.
			pidx = 64 * pidx;

			// Find the coefficient index in the c_lut row in order to determine
			// the linear index of the control point with respect to the current tile.
			for(m = 0; m < 64; m++) {
			    if(tex1Dfetch(tex_c_lut, pidx + m) == threadIdxInGrid) 
				break;
			}									
						
			// Accumulate the influence of each voxel in the current tile
						
			// To improve performance, we unroll the loop to operate 
			// on multiple voxels per iteration. An unrolling factor of four appears to be the best performer
						
			int unrolling_factor = 4; // Set this parameter to achieve the level of loop unrolling desired; could be 1 or 4
			int total_vox_per_rgn = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
			int modified_idx = (total_vox_per_rgn/unrolling_factor)*unrolling_factor; // The modified index is an integral multiple of the unrolling factor
			int lop_off = total_vox_per_rgn - modified_idx;
						
			for(qidx = 0; qidx < modified_idx; qidx = qidx + unrolling_factor) {
			    multiplier = tex1Dfetch(tex_q_lut, 64*qidx + m); // Voxel 1
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+1) + m); // Voxel 2
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+1) + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+2) + m); // Voxel 3
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+2) + 2) * multiplier;

			    multiplier = tex1Dfetch(tex_q_lut, 64*(qidx+3) + m); // Voxel 4
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*(qidx+3) + 2) * multiplier;
			}
						
			// Take care of any lop off voxels
			for(qidx = modified_idx; qidx < (modified_idx + lop_off); qidx++){
			    multiplier = tex1Dfetch(tex_q_lut, 64*qidx + m);
			    result.x += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 0) * multiplier;
			    result.y += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 1) * multiplier;
			    result.z += tex1Dfetch(tex_dc_dv, dc_dv_row + 3*qidx + 2) * multiplier;
			} 
		    } // if tile location is within the volume
		} // for each tile
	    } // 
	}
	grad[3*threadIdxInGrid+0] = result.x;
	grad[3*threadIdxInGrid+1] = result.y;
	grad[3*threadIdxInGrid+2] = result.z;
		
    }
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel1a
 *
 * This kernel fills the score stream.  It operates on the entire volume
 * at one time, and therefore the number of elements in the score stream
 * must be equal to the number of voxels in the volume.  The dc_dv
 * computations are contained in a separate kernel, 
 * bspline_cuda_score_e_mse_kernel1b, because they must be performed on
 * "set by set" basis.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel1a (
	float  *dc_dv,
	float  *score,
	float3 rdims,			// Number of tiles/regions in x, y, and z
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing)		// Dimensions of a single voxel (in mm)
{
	int3   vox_coordinate;	// X, Y, Z coordinates for this voxel	
	int3   p;				// Offset of the tile in the volume (x, y and z)
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	int3 displacement_in_vox_floor;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Get the X, Y, Z position of the voxel.
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);
	
		// Get the tile location of the voxel.
		p.x = vox_coordinate.x / vox_per_rgn.x;
		p.y = vox_coordinate.y / vox_per_rgn.y;
		p.z = vox_coordinate.z / vox_per_rgn.z;
	
		// Get the offset of the voxel within the tile.
		q.x = vox_coordinate.x - p.x * vox_per_rgn.x;
		q.y = vox_coordinate.y - p.y * vox_per_rgn.y;
		q.z = vox_coordinate.z - p.z * vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = qidx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}

			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
					// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = (int)(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = (int)(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = (int)(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = (diff * diff);
			}	
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel1b
 *
 * This kernel calculates the dc_dv values for a given "set."  Since
 * there are a total of 64 sets, this kernel must be executed 64 times 
 * using different sidx values to completely fill the dc_dv stream.  To
 * improve performance, the score calculations are contained in a
 * separate kernel, bspline_cuda_score_e_mse_kernel1a, and calculated 
 * for the entire volume at one time.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel1b (
	float  *dc_dv,
	float  *score,
	int3   sidx,			// Current "set index" given in x, y and z
	float3 rdims,			// Number of tiles/regions in x, y, and z
	int3   sdims,           // Dimensions of the set in tiles (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	int    total_vox_per_rgn,
	float3 pix_spacing)		// Dimensions of a single voxel (in mm)
{
	int3   s;				// Offset of the tile in the set (x, y and z)
	int3   p;				// Offset of the tile in the volume (x, y and z)
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Calculate the linear "set index," which is the index of the tile in the set that contains the 
	// voxel corresponding to this thread.
	int tileIdxInSet = threadIdxInGrid / total_vox_per_rgn;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the offset of the tile within the set in the x, y, and z directions.
		s.x = tileIdxInSet % sdims.x;
		s.y = ((tileIdxInSet - s.x) / sdims.x) % sdims.y;
		s.z = ((((tileIdxInSet - s.x) / sdims.x) - s.y) / sdims.y) % sdims.z;

		// Calculate the offset of the tile in the volume, based on the set offset.
		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = qidx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
					// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel1
 *
 * As an alternative to using bspline_cuda_score_e_mse_kernel1a and
 * bspline_cuda_score_e_mse_kernel1b separately, this kernel computes
 * both the score and dc_dv stream values on a set by set basis.  Since
 * there are a total of 64 sets, this kernel must be executed 64 times 
 * using different sidx values to completely fill the score and dc_dv 
 * streams.  The performance is worse than using kernel1a and kernel1b.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel1 (
	float  *dc_dv,
	float  *score,
	int3   sidx,			// Current "set index" given in x, y and z
	float3 rdims,			// Number of tiles/regions in x, y, and z
	int3   sdims,           // Dimensions of the set in tiles (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	int    total_vox_per_rgn,
	float3 pix_spacing)		// Dimensions of a single voxel (in mm)
{
	int3   s;				// Offset of the tile in the set (x, y and z)
	int3   p;				// Offset of the tile in the volume (x, y and z)
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Calculate the linear "set index," which is the index of the tile in the set that contains the 
	// voxel corresponding to this thread.
	int tileIdxInSet = threadIdxInGrid / total_vox_per_rgn;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Calculate the offset of the tile within the set in the x, y, and z directions.
		s.x = tileIdxInSet % sdims.x;
		s.y = ((tileIdxInSet - s.x) / sdims.x) % sdims.y;
		s.z = ((((tileIdxInSet - s.x) / sdims.x) - s.y) / sdims.y) % sdims.z;

		// Calculate the offset of the tile in the volume, based on the set offset.
		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = qidx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
					// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				// The score calculation has been moved to bspline_cuda_score_e_kernel1a.
				score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel2_by_sets
 *
 * This version of kernel2 updates the gradient stream for a given set
 * of tiles.  It performs the calculation for the entire set at once,
 * which improves parallelism and therefore improves performance as
 * compared to the tile by tile implementation, which is found below.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel2_by_sets(
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int3   sidx,
	int3   sdims,
	float3 rdims,
	int3   vox_per_rgn,
	int    threads_per_tile,
	int    num_threads)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Calculate the linear "set index," which is the index of the tile in the set that contains the 
	// voxel corresponding to this thread.
	int tileIdxInSet = threadIdxInGrid / threads_per_tile;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int3 s; // Offset of the tile in the set (x, y, and z)
		int3 p; // Offset of the tile in the volume (x, y, and z)
		int m;
		int num_vox;
		int xyzOffset;
		int tileOffset;
		int pidx;
		int cidx;
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Calculate the offset of the tile within the set in the x, y, and z directions.
		s.x = tileIdxInSet % sdims.x;
		s.y = ((tileIdxInSet - s.x) / sdims.x) % sdims.y;
		s.z = ((((tileIdxInSet - s.x) / sdims.x) - s.y) / sdims.y) % sdims.z;

		// Calculate the offset of the tile in the volume, based on the set offset.
		p.x = (s.x * 4) + sidx.x;
		p.y = (s.y * 4) + sidx.y;
		p.z = (s.z * 4) + sidx.z;

		// Use the offset of the tile in the volume to compute the index into the c_lut.
		pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;

		// Calculate the linear index of the control point in the range [0, 63].
		m = (threadIdxInGrid % threads_per_tile) / 3;

		// Determine if this thread corresponds to the x, y, or z coordinate,
		// where x = 0, y = 1, and z = 2.
		xyzOffset = (threadIdxInGrid % threads_per_tile) - (m * 3);

		// Calculate the index into the coefficient lookup table.
		cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels per tile.
		num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		// Calculate the offset of this tile in the dc_dv array.
		tileOffset = 3 * num_vox * tileIdxInSet;

		for(qidx = 0; qidx < num_vox - 8; qidx = qidx + 8) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+7 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+6 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+5 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+4 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+3 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+2 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+1 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel2_by_tiles
 * This version of kernel2 updates the gradient stream for a given tile.
 * Since it operates on only one tile in a set at a given time, the
 * performance is worse than bspline_cuda_score_e_mse_kernel2_by_sets.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel2_by_tiles (
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int    num_threads,
	int3   p,
	float3 rdims,
	int    offset,
	int3   vox_per_rgn,
	int    total_vox_per_rgn) // Volume of a tile in voxels)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int m;
		int num_vox;
		int xyzOffset;
		int tileOffset;
		int cidx;
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		
		// Calculate the linear index of the control point.
		m = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		xyzOffset = threadIdxInGrid - (m * 3);

		// Calculate index into coefficient texture.
		cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels in the region.
		num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		// Calculate the offset of this tile in the dc_dv array.
		tileOffset = 3 * num_vox * offset;

		/* ORIGINAL CODE: Looked at each offset serially.
		// Serial across offsets.
		for(int qidx = 0; qidx < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z); qidx++) {
			result += tex1Dfetch(tex_dc_dv, 3*qidx + offset) * tex1Dfetch(tex_q_lut, 64*qidx + m);
		}
		*/

		// NAGA: Unrolling the loop 8 times; 4 seems to work as well as 8
		// FOR_CHRIS: FIX to make sure the unrolling works with an arbitrary loop index
		for(qidx = 0; qidx < num_vox - 8; qidx = qidx + 8) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+7 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+6 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+5 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+4 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+3 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+2 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+1 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
	}
}

/***********************************************************************
 * bspline_cuda_score_e_mse_kernel2_by_tiles_v2
 *
 * In comparison to bspline_cuda_score_e_mse_kernel2_by_tiles_v2, this
 * kernel uses multiple threads to accumulate the influence from a tile.
 * The threads are synchronized at the end so that the partial sums can
 * be exchanged using shared memory, summed together, and saved to the
 * gradient stream.  The number of threads being used for each control
 * point must be given as an argument.  The performance is better than
 * bspline_cuda_score_e_mse_kernel2_by_tiles_v2, but the implementation
 * is still buggy.
 ***********************************************************************/
__global__ void bspline_cuda_score_e_mse_kernel2_by_tiles_v2 (
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int    num_threads,
	int3   p,
	float3 rdims,
	int    offset,
	int3   vox_per_rgn,
	int    threadsPerControlPoint)
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

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Set the number of threads being used to work on each control point.
		int tpcp = threadsPerControlPoint;

		// Calculate the linear index of the control point.
		int m = threadIdxInGrid / (threadsPerControlPoint * 3);

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		int xyzOffset = (threadIdxInGrid / threadsPerControlPoint) - (m * 3);

		// Determine the thread offset for this control point, in the range [0, threadsPerControlPoint).
		int cpThreadOffset = threadIdxInGrid % threadsPerControlPoint;

		// Calculate index into coefficient texture.
		int cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels in the region.
		int num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		// Calculate the offset of this tile in the dc_dv array.
		int tileOffset = 3 * num_vox * offset;

		for(qidx = cpThreadOffset; qidx < num_vox - (8*tpcp); qidx = qidx + (8*tpcp)) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+(7*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+(6*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+(5*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+(4*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+(3*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+(2*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+(1*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + tileOffset + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		sdata[(tpcp * threadIdxInBlock) + cpThreadOffset] = result;
		
		// Wait for the other threads in the thread block to reach this point.
		__syncthreads();

		if(cpThreadOffset == 0) {
			result = sdata[(tpcp * threadIdxInBlock) + 0] + sdata[(tpcp * threadIdxInBlock) + 1];
				
			/*
			result = 0.0;

			// Accumulate all the partial results for this control point.
			for(int i = 0; i < tpcp; i++) {
				result += sdata[(tpcp * threadIdxInBlock) + i];
			}
			*/

			// Update the gradient stream.
			grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
		}			
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel1
 *
 * This kernel is one of two used in the CUDA implementation of 
 * score_d_mse, which is intended to have reduced memory requirements.  
 * It calculuates the score and dc_dv values on a region by region basis 
 * rather than for the entire volume at once.  As a result, the score 
 * stream need only to have as many elements as there are voxels in a 
 * region.  When executing this kernel, the number of threads should be 
 * close to (but greater than) the number of voxels in a region.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel1 (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	int3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInGrid < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = threadIdxInGrid % vox_per_rgn.x;
		q.y = ((threadIdxInGrid - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((threadIdxInGrid - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;

			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			// qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = threadIdxInGrid * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// diff = 0.0;
				// valid = 0;
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = (int)(displacement_in_vox.x);
				displacement_in_vox_round.x = rintf(displacement_in_vox.x);	// Single instruction round
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = (int)(displacement_in_vox.y);
				displacement_in_vox_round.y = rintf(displacement_in_vox.y);	// Single instruction round
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = (int)(displacement_in_vox.z);
				displacement_in_vox_round.z = rintf(displacement_in_vox.z);	// Single instruction round
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				score[threadIdxInGrid] = tex1Dfetch(tex_score, threadIdxInGrid) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[3*(threadIdxInGrid)+0] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0);
				dc_dv[3*(threadIdxInGrid)+1] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1);
				dc_dv[3*(threadIdxInGrid)+2] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel1_v2
 *
 * This kernel is one of two used in the CUDA implementation of 
 * score_d_mse, which is intended to have reduced memory requirements.  
 * It calculuates the score and dc_dv values on a region by region basis 
 * rather than for the entire volume at once.  As a result, the score 
 * stream need only to have as many elements as there are voxels in a 
 * region.  When executing this kernel, the number of threads should be 
 * close to (but greater than) the number of voxels in a region.
 *
 * In comparison to bspline_cuda_score_d_mse_kernel1, this kernel 
 * computes the x, y, and z portions of each value in separate threads 
 * for increased parallelism.  The performance is worse than 
 * bspline_cuda_score_d_mse_kernel1, so this version should not be used.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel1_v2 (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.

	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;

	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	int lridx = 0;  // Linear index within the region
	int offset = 0; // x = 0, y = 1, z = 2

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInGrid < (3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the linear index of the voxel in the region. Will be in the range
		// (0, vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z - 1).
		lridx = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (lridx * 3);		

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = lridx % vox_per_rgn.x;
		q.y = ((lridx - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((lridx - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel in the volume.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;
			
			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			// qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = lridx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				// Do nothing.
			}
			else {

				//-----------------------------------------------------------------
				// Compute interpolation fractions.
				//-----------------------------------------------------------------

				// Clamp and interpolate along the X axis.
				displacement_in_vox_floor.x = floor(displacement_in_vox.x);
				displacement_in_vox_round.x = round(displacement_in_vox.x);
				fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
				if(displacement_in_vox_floor.x < 0){
					displacement_in_vox_floor.x = 0;
					displacement_in_vox_round.x = 0;
					fx2 = 0.0;
				}
				else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
					displacement_in_vox_floor.x = volume_dim.x - 2;
					displacement_in_vox_round.x = volume_dim.x - 1;
					fx2 = 1.0;
				}
				fx1 = 1.0 - fx2;

				// Clamp and interpolate along the Y axis.
				displacement_in_vox_floor.y = floor(displacement_in_vox.y);
				displacement_in_vox_round.y = round(displacement_in_vox.y);
				fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
				if(displacement_in_vox_floor.y < 0){
					displacement_in_vox_floor.y = 0;
					displacement_in_vox_round.y = 0;
					fy2 = 0.0;
				}
				else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
					displacement_in_vox_floor.y = volume_dim.y - 2;
					displacement_in_vox_round.y = volume_dim.y - 1;
					fy2 = 1.0;
				}
				fy1 = 1.0 - fy2;
				
				// Clamp and intepolate along the Z axis.
				displacement_in_vox_floor.z = floor(displacement_in_vox.z);
				displacement_in_vox_round.z = round(displacement_in_vox.z);
				fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
				if(displacement_in_vox_floor.z < 0){
					displacement_in_vox_floor.z = 0;
					displacement_in_vox_round.z = 0;
					fz2 = 0.0;
				}
				else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
					displacement_in_vox_floor.z = volume_dim.z - 2;
					displacement_in_vox_round.z = volume_dim.z - 1;
					fz2 = 1.0;
				}
				fz1 = 1.0 - fz2;
				
				//-----------------------------------------------------------------
				// Compute moving image intensity using linear interpolation.
				//-----------------------------------------------------------------

				mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
				m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
				m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
				m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
				m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
				m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
				m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
				m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
				m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
				m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

				//-----------------------------------------------------------------
				// Compute intensity difference.
				//-----------------------------------------------------------------

				// diff[threadIdxInGrid] = fixed_image[threadIdxInGrid] - m_val;
				diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
				
				//-----------------------------------------------------------------
				// Accumulate the score.
				//-----------------------------------------------------------------

				if(offset == 0)
					score[lridx] = tex1Dfetch(tex_score, lridx) + (diff * diff);

				//-----------------------------------------------------------------
				// Compute dc_dv for this offset
				//-----------------------------------------------------------------
				
				// Compute spatial gradient using nearest neighbors.
				mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
				dc_dv[threadIdxInGrid] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + offset);
			}		
		}
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel1_v3
 *
 * This kernel is one of two used in the CUDA implementation of 
 * score_d_mse, which is intended to have reduced memory requirements.  
 * It calculuates the score and dc_dv values on a region by region basis 
 * rather than for the entire volume at once.  As a result, the score 
 * stream need only to have as many elements as there are voxels in a 
 * region.  When executing this kernel, the number of threads should be 
 * close to (but greater than) the number of voxels in a region.
 *
 * In comparison to bspline_cuda_score_d_mse_kernel2, this kernel uses 
 * shared memory to exchange data between threads to reduce the number
 * of memory accesses.  The performance is worse than 
 * bspline_cuda_score_d_mse_kernel1, so this version should not be used.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel1_v3 (
	float  *dc_dv,
	float  *score,			
	int3   p,				// Offset of the tile in the volume (x, y and z)
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// Image origin (in mm)
    float3 img_spacing,     // Image spacing (in mm)
	float3 img_offset,		// Offset corresponding to the region of interest
    int3   roi_offset,	    // Position of first vox in ROI (in vox)
    int3   roi_dim,			// Dimension of ROI (in vox)
    int3   vox_per_rgn,	    // Knot spacing (in vox)
	float3 pix_spacing,		// Dimensions of a single voxel (in mm)
	float3 rdims)			// # of regions in (x,y,z)
{
	// Shared memory is allocated on a per block basis.  Therefore, only allocate 
	// (sizeof(data) * blocksize) memory when calling the kernel.
	extern __shared__ float sdata[]; 

	int lridx = 0;  // Linear index within the region
	int offset = 0; // x = 0, y = 1, z = 2

	int3   q;				// Offset within the tile (measured in voxels).
	int3   coord_in_volume;	// Offset within the volume (measured in voxels).
	int    fv;				// Index of voxel in linear image array.
	int    pidx;			// Index into c_lut.
	int    qidx;			// Index into q_lut.
	int    cidx;			// Index into the coefficient table.
	float  P;				
	float3 N;				// Multiplier values.		
	float3 d;				// B-spline deformation vector.
	float  diff;
	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  mvr;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Calculate the number of unusable threads in each block.
	int threadsLostPerBlock = threadsPerBlock - (threadsPerBlock / 3) * 3;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * (threadsPerBlock - threadsLostPerBlock)) + threadIdxInBlock;

	// Set the "write flag" to 0.
	sdata[2*(threadIdxInBlock/3)+2] = 0.0;

	// If the voxel lies outside the region, do nothing.
	if(threadIdxInBlock < (threadsPerBlock - threadsLostPerBlock) &&
		threadIdxInGrid < (3 * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z))
	{	
		// Calculate the linear index of the voxel in the region. Will be in the range
		// (0, vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z - 1).
		lridx = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (lridx * 3);		

		// Only one out of every three threads needs to calculate the following information.
		// All other threads get the data from shared memory.
		if(offset ==  0) {

		// Calculate the x, y and z offsets of the voxel within the tile.
		q.x = lridx % vox_per_rgn.x;
		q.y = ((lridx - q.x) / vox_per_rgn.x) % vox_per_rgn.y;
		q.z = ((((lridx - q.x) / vox_per_rgn.x) - q.y) / vox_per_rgn.y) % vox_per_rgn.z;

		// Calculate the x, y and z offsets of the voxel within the volume.
		coord_in_volume.x = roi_offset.x + p.x * vox_per_rgn.x + q.x;
		coord_in_volume.y = roi_offset.y + p.y * vox_per_rgn.y + q.y;
		coord_in_volume.z = roi_offset.z + p.z * vox_per_rgn.z + q.z;

		// If the voxel lies outside the image, do nothing.
		if(coord_in_volume.x <= (roi_offset.x + roi_dim.x) || 
			coord_in_volume.y <= (roi_offset.y + roi_dim.y) ||
			coord_in_volume.z <= (roi_offset.z + roi_dim.z)) {

			// Compute the linear index of fixed image voxel in the volume.
			fv = (coord_in_volume.z * volume_dim.x * volume_dim.y) + (coord_in_volume.y * volume_dim.x) + coord_in_volume.x;
			
			//-----------------------------------------------------------------
			// Calculate the B-Spline deformation vector.
			//-----------------------------------------------------------------

			// Use the offset of the voxel within the region to compute the index into the c_lut.
			pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
			pidx = pidx * 64;

			// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
			// qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
			qidx = lridx * 64;

			// Compute the deformation vector.
			d.x = 0.0;
			d.y = 0.0;
			d.z = 0.0;

			for(int k = 0; k < 64; k++)
			{
				// Calculate the index into the coefficients array.
				cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
				
				// Fetch the values for P, Ni, Nj, and Nk.
				P   = tex1Dfetch(tex_q_lut, qidx + k); 
				N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
				N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
				N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

				// Update the output (v) values.
				d.x += P * N.x;
				d.y += P * N.y;
				d.z += P * N.z;
			}
			
			//-----------------------------------------------------------------
			// Find correspondence in the moving image.
			//-----------------------------------------------------------------

			// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
			distance_from_image_origin.x = img_origin.x + (pix_spacing.x * coord_in_volume.x);
			distance_from_image_origin.y = img_origin.y + (pix_spacing.y * coord_in_volume.y);
			distance_from_image_origin.z = img_origin.z + (pix_spacing.z * coord_in_volume.z);
			
			// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
			displacement_in_mm.x = distance_from_image_origin.x + d.x;
			displacement_in_mm.y = distance_from_image_origin.y + d.y;
			displacement_in_mm.z = distance_from_image_origin.z + d.z;

			// Calculate the displacement value in terms of voxels.
			displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
			displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
			displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

			// Check if the displaced voxel lies outside the region of interest.
			if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
				(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
				(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
				
				if(offset == 0) {
					sdata[2*(threadIdxInBlock/3)] = 0.0;
					sdata[2*(threadIdxInBlock/3)+1] = 0.0;
				}
			}
			else {
					
					//-----------------------------------------------------------------
					// Compute interpolation fractions.
					//-----------------------------------------------------------------

					// Clamp and interpolate along the X axis.
					displacement_in_vox_floor.x = floor(displacement_in_vox.x);
					displacement_in_vox_round.x = round(displacement_in_vox.x);
					fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
					if(displacement_in_vox_floor.x < 0){
						displacement_in_vox_floor.x = 0;
						displacement_in_vox_round.x = 0;
						fx2 = 0.0;
					}
					else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
						displacement_in_vox_floor.x = volume_dim.x - 2;
						displacement_in_vox_round.x = volume_dim.x - 1;
						fx2 = 1.0;
					}
					fx1 = 1.0 - fx2;

					// Clamp and interpolate along the Y axis.
					displacement_in_vox_floor.y = floor(displacement_in_vox.y);
					displacement_in_vox_round.y = round(displacement_in_vox.y);
					fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
					if(displacement_in_vox_floor.y < 0){
						displacement_in_vox_floor.y = 0;
						displacement_in_vox_round.y = 0;
						fy2 = 0.0;
					}
					else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
						displacement_in_vox_floor.y = volume_dim.y - 2;
						displacement_in_vox_round.y = volume_dim.y - 1;
						fy2 = 1.0;
					}
					fy1 = 1.0 - fy2;
					
					// Clamp and intepolate along the Z axis.
					displacement_in_vox_floor.z = floor(displacement_in_vox.z);
					displacement_in_vox_round.z = round(displacement_in_vox.z);
					fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
					if(displacement_in_vox_floor.z < 0){
						displacement_in_vox_floor.z = 0;
						displacement_in_vox_round.z = 0;
						fz2 = 0.0;
					}
					else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
						displacement_in_vox_floor.z = volume_dim.z - 2;
						displacement_in_vox_round.z = volume_dim.z - 1;
						fz2 = 1.0;
					}
					fz1 = 1.0 - fz2;
					
					//-----------------------------------------------------------------
					// Compute moving image intensity using linear interpolation.
					//-----------------------------------------------------------------

					mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
					m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
					m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
					m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
					m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
					m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
					m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
					m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
					m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
					m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

					//-----------------------------------------------------------------
					// Compute intensity difference.
					//-----------------------------------------------------------------

					diff = tex1Dfetch(tex_fixed_image, fv) - m_val;
					
					//-----------------------------------------------------------------
					// Accumulate the score.
					//-----------------------------------------------------------------
				
					score[lridx] = tex1Dfetch(tex_score, lridx) + (diff * diff);

					//-----------------------------------------------------------------
					// Compute dc_dv for this offset
					//-----------------------------------------------------------------
					
					// Compute spatial gradient using nearest neighbors.
					mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;

					// Store this data in shared memory.
					sdata[2*(threadIdxInBlock/3)] = diff;
					sdata[2*(threadIdxInBlock/3)+1] = mvr;
					sdata[2*(threadIdxInBlock/3)+2] = 1.0;
				}				
			}
		}
	}

	// Wait until all the threads in this thread block reach this point.
	__syncthreads();

	// dc_dv[threadIdxInGrid] = diff * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + offset);

	if(sdata[2*(threadIdxInBlock/3)+2] == 1.0)
		dc_dv[threadIdxInGrid] = sdata[2*(threadIdxInBlock/3)] * 
			tex1Dfetch(tex_moving_grad, (3 * (int)sdata[2*(threadIdxInBlock/3)+1]) + offset);
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel2
 *
 * This kernel is the second of two used in the CUDA implementation of
 * score_d_mse.  It calculates the values for the gradient stream on 
 * a tile by tile basis.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel2 (
	float  *dc_dv,
	float  *grad,
	float  *gpu_q_lut,
	int    num_threads,
	int3   p,
	float3 rdims,
	int3   vox_per_rgn)
{
	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{	
		int m;
		int offset;
		int cidx;
		int qidx;
		int num_vox;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		
		// Calculate the linear index of the control point.
		m = threadIdxInGrid / 3;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		offset = threadIdxInGrid - (m * 3);

		// Calculate index into coefficient texture.
		cidx = tex1Dfetch(tex_c_lut, 64*pidx + m) * 3;

		// Calculate the number of voxels in the region.
		num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		/* ORIGINAL CODE: Looked at each offset serially.
		// Serial across offsets.
		for(int qidx = 0; qidx < (vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z); qidx++) {
			result += tex1Dfetch(tex_dc_dv, 3*qidx + offset) * tex1Dfetch(tex_q_lut, 64*qidx + m);
		}
		*/

		// NAGA: Unrolling the loop 8 times; 4 seems to work as well as 8
		// FOR_CHRIS: FIX to make sure the unrolling works with an arbitrary loop index
		for(qidx = 0; qidx < num_vox - 8; qidx = qidx + 8) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+7 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+7) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+7) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+6 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+6) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+6) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+5 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+5) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+5) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+4 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+4) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+4) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+3 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+3) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+3) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+2 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+2) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+2) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+1 < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)   + offset) * tex1Dfetch(tex_q_lut, 64*(qidx)   + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+1) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx+1) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			result += tex1Dfetch(tex_dc_dv, 3*(qidx) + offset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		grad[cidx + offset] = tex1Dfetch(tex_grad, cidx + offset) + result;
	}
}

/***********************************************************************
 * bspline_cuda_score_d_mse_kernel2_v2
 *
 * This kernel is the second of two used in the CUDA implementation of
 * score_d_mse.  It calculates the values for the gradient stream on 
 * a tile by tile basis.
 *
 * In comparison to bspline_cuda_score_d_mse_kernel2, this kernel uses
 * multiple threads to accumulate the influence from a tile.  The
 * threads are synchronized at the end so that the partial sums can be
 * exchanged using shared memory, totaled, and accumulated into the
 * gradient stream.  The number of threads being used for each control
 * point must be given as an argument.  The performance is better than 
 * bspline_cuda_score_d_mse_kernel2, although the implementation is
 * still buggy.
 ***********************************************************************/
__global__ void bspline_cuda_score_d_mse_kernel2_v2 (
	float* grad,
	int    num_threads,
	int3   p,
	float3 rdims,
	int3   vox_per_rgn,
	int    threadsPerControlPoint)
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

	// If the thread does not correspond to a control point, do nothing.
	if(threadIdxInGrid < num_threads)
	{
		int qidx;
		float result = 0.0;
		float temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;

		// Set the number of threads being used to work on each control point.
		int tpcp = threadsPerControlPoint;

		// Calculate the linear index of the control point.
		int m = threadIdxInGrid / (threadsPerControlPoint * 3);

		// Use the offset of the voxel within the region to compute the index into the c_lut.
		int pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;

		// Calculate the coordinate offset (x = 0, y = 1, z = 2).
		int xyzOffset = (threadIdxInGrid / threadsPerControlPoint) - (m * 3);

		// Determine the thread offset for this control point, in the range [0, threadsPerControlPoint).
		int cpThreadOffset = threadIdxInGrid % threadsPerControlPoint;

		// Calculate index into coefficient texture.
		int cidx = tex1Dfetch(tex_c_lut, 64 * pidx + m) * 3;

		// Calculate the number of voxels in the region.
		int num_vox = vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;

		for(qidx = cpThreadOffset; qidx < num_vox - (8*tpcp); qidx = qidx + (8*tpcp)) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		
		if(qidx+(7*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			temp7 = tex1Dfetch(tex_dc_dv, 3*(qidx+(7*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(7*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6 + temp7;
		}
		else if(qidx+(6*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			temp6 = tex1Dfetch(tex_dc_dv, 3*(qidx+(6*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(6*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6;
		}
		else if(qidx+(5*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			temp5 = tex1Dfetch(tex_dc_dv, 3*(qidx+(5*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(5*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}
		else if(qidx+(4*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			temp4 = tex1Dfetch(tex_dc_dv, 3*(qidx+(4*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(4*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3 + temp4;
		}
		else if(qidx+(3*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			temp3 = tex1Dfetch(tex_dc_dv, 3*(qidx+(3*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(3*tpcp)) + m);
			result += temp0 + temp1 + temp2 + temp3;
		}
		else if(qidx+(2*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			temp2 = tex1Dfetch(tex_dc_dv, 3*(qidx+(2*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(2*tpcp)) + m);
			result += temp0 + temp1 + temp2;
		}
		else if(qidx+(1*tpcp) < num_vox) {
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);
			temp1 = tex1Dfetch(tex_dc_dv, 3*(qidx+(1*tpcp)) + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx+(1*tpcp)) + m);
			result += temp0 + temp1;
		}
		else if(qidx < num_vox)
			temp0 = tex1Dfetch(tex_dc_dv, 3*(qidx)          + xyzOffset) * tex1Dfetch(tex_q_lut, 64*(qidx) + m);

		sdata[(tpcp * threadIdxInBlock) + cpThreadOffset] = result;
		
		// Wait for the other threads in the thread block to reach this point.
		__syncthreads();

		if(cpThreadOffset == 0) {
			result = 0.0;

			// Accumulate all the partial results for this control point.
			for(int i = 0; i < tpcp; i++) {
				result += sdata[(tpcp * threadIdxInBlock) + i];
			}
			
			// Update the gradient stream.
			grad[cidx + xyzOffset] = tex1Dfetch(tex_grad, cidx + xyzOffset) + result;
		}			
	}
}

/***********************************************************************
 * bspline_cuda_compute_dxyz_kernel
 *
 * This kernel computes the displacement values in the x, y, and 
 * z directions.
 ***********************************************************************/
__global__ void bspline_cuda_compute_dxyz_kernel(
	int   *c_lut,
	float *q_lut,
	float *coeff,
	int3 volume_dim,
	int3 vox_per_rgn,
	float3 rdims,
	float *dx,
	float *dy,
	float *dz
	)
{
	int3 vox_coordinate;	// X, Y, Z coordinates for this voxel	
	int3 p;				    // Tile index.
	int3 q;				    // Offset within tile.
	int pidx;				// Index into c_lut.
	int qidx;				// Index into q_lut.
	int cidx;				// Index into the coefficient table.
	int* prow;				// First element in the correct row in c_lut.
	float P;				
	float3 N;				// Multiplier values.		
	float3 output;			// Output values.

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// If the voxel lies outside the volume, do nothing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{
		// Get the X, Y, Z position of the voxel.
		// vox_coordinate.z = floor(threadIdxInGrid / (volume_dim.x * volume_dim.y));
		// vox_coordinate.y = floor((threadIdxInGrid - vox_coordinate.z * (volume_dim.x * volume_dim.y)) / volume_dim.x);
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);
			
		// Get the tile location of the voxel.
		p.x = vox_coordinate.x / vox_per_rgn.x;
		p.y = vox_coordinate.y / vox_per_rgn.y;
		p.z = vox_coordinate.z / vox_per_rgn.z;
				
		// Get the offset of the voxel within the tile.
		q.x = vox_coordinate.x - p.x * vox_per_rgn.x;
		q.y = vox_coordinate.y - p.y * vox_per_rgn.y;
		q.z = vox_coordinate.z - p.z * vox_per_rgn.z;
				
		// Use the tile location of the voxel to compute the index into the c_lut.
		pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		prow = &c_lut[pidx*64];
		pidx = pidx * 64;

		// Use the offset of the voxel to compute the index into the multiplier LUT or q_lut.
		qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
		// qrow = &q_lut[qidx*64];
		qidx = qidx * 64;

		// Initialize output values.
		output.x = 0.0;
		output.y = 0.0;
		output.z = 0.0;

		for(int k = 0; k < 64; k++)
		{
			// Calculate the index into the coefficients array.
			cidx = 3 * prow[k];
			// cidx = 3 * tex1Dfetch(tex_c_lut, pidx + k); 
			
			// Fetch the values for P, Ni, Nj, and Nk.
			// P = qrow[k];
			P  = tex1Dfetch(tex_q_lut, qidx + k); 
			N.x = tex1Dfetch(tex_coeff, cidx + 0);  // x-value
			N.y = tex1Dfetch(tex_coeff, cidx + 1);  // y-value
			N.z = tex1Dfetch(tex_coeff, cidx + 2);  // z-value

			// Update the output (v) values.
			output.x += P * N.x;
			output.y += P * N.y;
			output.z += P * N.z;
		}

		// Save the calculated values to the output streams.
		dx[threadIdxInGrid] = output.x;
		dy[threadIdxInGrid] = output.y;
		dz[threadIdxInGrid] = output.z;
	}
}


/***********************************************************************
 * bspline_cuda_compute_diff_kernel
 *
 * This kernel computes the intensity difference between the voxels
 * in the moving and fixed images.
 ***********************************************************************/
__global__ void bspline_cuda_compute_diff_kernel (
	float* fixed_image,
	float* moving_image,
	float* dx,
	float* dy,
	float* dz,
	float* diff,
	int*   valid_voxels,
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	float3 img_origin,		// x, y, z coordinates for the image origin
	float3 pix_spacing,		// Dimensions of a single voxel in millimeters
	float3 img_offset)		// Offset corresponding to the region of interest
{	

	int3   vox_coordinate;
	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	int3   displacement_in_vox_floor;
	float  fx1, fx2, fy1, fy2, fz1, fz2;
	int    mvf;
	float  m_val;
	float  m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Ensure that the thread index corresponds to a voxel in the volume before continuing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{ 
		// Get the x, y, z position of the voxel.
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);

		// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
		distance_from_image_origin.x = img_origin.x + (pix_spacing.x * vox_coordinate.x);
		distance_from_image_origin.y = img_origin.y + (pix_spacing.y * vox_coordinate.y);
		distance_from_image_origin.z = img_origin.z + (pix_spacing.z * vox_coordinate.z);
		
		// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
		displacement_in_mm.x = distance_from_image_origin.x + tex1Dfetch(tex_dx, threadIdxInGrid); //dx[threadIdxInGrid];
		displacement_in_mm.y = distance_from_image_origin.y + tex1Dfetch(tex_dy, threadIdxInGrid); //dy[threadIdxInGrid];
		displacement_in_mm.z = distance_from_image_origin.z + tex1Dfetch(tex_dz, threadIdxInGrid); //dz[threadIdxInGrid];

		// Calculate the displacement value in terms of voxels.
		displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
		displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
		displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

		// Check if the displaced voxel lies outside the region of interest.
		if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
			(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
			(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
			diff[threadIdxInGrid] = 0.0;
			valid_voxels[threadIdxInGrid] = 0;
		}
		else {

			// Clamp and interpolate along the X axis.
			displacement_in_vox_floor.x = (int)floor(displacement_in_vox.x);
			fx2 = displacement_in_vox.x - displacement_in_vox_floor.x;
			if(displacement_in_vox_floor.x < 0){
				displacement_in_vox_floor.x = 0;
				fx2 = 0.0;
			}
			else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
				displacement_in_vox_floor.x = volume_dim.x - 2;
				fx2 = 1.0;
			}
			fx1 = 1.0 - fx2;
			
			// Clamp and interpolate along the Y axis.
			displacement_in_vox_floor.y = (int)floor(displacement_in_vox.y);
			fy2 = displacement_in_vox.y - displacement_in_vox_floor.y;
			if(displacement_in_vox_floor.y < 0){
				displacement_in_vox_floor.y = 0;
				fy2 = 0.0;
			}
			else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
				displacement_in_vox_floor.y = volume_dim.y - 2;
				fy2 = 1.0;
			}
			fy1 = 1.0 - fy2;
			
			// Clamp and intepolate along the Z axis.
			displacement_in_vox_floor.z = (int)floor(displacement_in_vox.z);
			fz2 = displacement_in_vox.z - displacement_in_vox_floor.z;
			if(displacement_in_vox_floor.z < 0){
				displacement_in_vox_floor.z = 0;
				fz2 = 0.0;
			}
			else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
				displacement_in_vox_floor.z = volume_dim.z - 2;
				fz2 = 1.0;
			}
			fz1 = 1.0 - fz2;
			
			// Compute moving image intensity using linear interpolation.
			mvf = (displacement_in_vox_floor.z * volume_dim.y + displacement_in_vox_floor.y) * volume_dim.x + displacement_in_vox_floor.x;
			/*
			m_x1y1z1 = fx1 * fy1 * fz1 * moving_image[mvf];
			m_x2y1z1 = fx2 * fy1 * fz1 * moving_image[mvf + 1];
			m_x1y2z1 = fx1 * fy2 * fz1 * moving_image[mvf + volume_dim.x];
			m_x2y2z1 = fx2 * fy2 * fz1 * moving_image[mvf + volume_dim.x + 1];
			m_x1y1z2 = fx1 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x];
			m_x2y1z2 = fx2 * fy1 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + 1];
			m_x1y2z2 = fx1 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x];
			m_x2y2z2 = fx2 * fy2 * fz2 * moving_image[mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1];
			*/
			m_x1y1z1 = fx1 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf);
			m_x2y1z1 = fx2 * fy1 * fz1 * tex1Dfetch(tex_moving_image, mvf + 1);
			m_x1y2z1 = fx1 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x);
			m_x2y2z1 = fx2 * fy2 * fz1 * tex1Dfetch(tex_moving_image, mvf + volume_dim.x + 1);
			m_x1y1z2 = fx1 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x);
			m_x2y1z2 = fx2 * fy1 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + 1);
			m_x1y2z2 = fx1 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x);
			m_x2y2z2 = fx2 * fy2 * fz2 * tex1Dfetch(tex_moving_image, mvf + volume_dim.y * volume_dim.x + volume_dim.x + 1);
			m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 + m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

			// Compute intensity difference.
			// diff[threadIdxInGrid] = fixed_image[threadIdxInGrid] - m_val;
			diff[threadIdxInGrid] = tex1Dfetch(tex_fixed_image, threadIdxInGrid) - m_val;
			valid_voxels[threadIdxInGrid] = 1;
		}
	}
}

/***********************************************************************
 * bspline_cuda_compute_dc_dv_kernel
 *
 * This kernel computes the dc_dv values used to update the control knot
 * coefficients.
 ***********************************************************************/
__global__ void bspline_cuda_compute_dc_dv_kernel (
	float  *fixed_image,
	float  *moving_image,
	float  *moving_grad,
	int    *c_lut,
	float  *q_lut,
	float  *dx,
	float  *dy,
	float  *dz,
	float  *diff,
	float  *dc_dv_x,
	float  *dc_dv_y,
	float  *dc_dv_z,
	// float  *grad,
	int    *valid_voxels,
	int3   volume_dim,		// x, y, z dimensions of the volume in voxels
	int3   vox_per_rgn,
	float3 rdims,
	float3 img_origin,		// x, y, z coordinates for the image origin
	float3 pix_spacing,		// Dimensions of a single voxel in millimeters
	float3 img_offset)		// Offset corresponding to the region of interest
{	
	int3   vox_coordinate;
	float3 distance_from_image_origin;
	float3 displacement_in_mm; 
	float3 displacement_in_vox;
	float3 displacement_in_vox_floor;
	float3 displacement_in_vox_round;
	float  mvr;

	// Calculate the index of the thread block in the grid.
	int blockIdxInGrid  = (gridDim.x * blockIdx.y) + blockIdx.x;

	// Calculate the total number of threads in each thread block.
	int threadsPerBlock  = (blockDim.x * blockDim.y * blockDim.z);

	// Next, calculate the index of the thread in its thread block, in the range 0 to threadsPerBlock.
	int threadIdxInBlock = (blockDim.x * blockDim.y * threadIdx.z) + (blockDim.x * threadIdx.y) + threadIdx.x;

	// Finally, calculate the index of the thread in the grid, based on the location of the block in the grid.
	int threadIdxInGrid = (blockIdxInGrid * threadsPerBlock) + threadIdxInBlock;

	// Ensure that the thread index corresponds to a voxel in the volume before continuing.
	if(threadIdxInGrid < (volume_dim.x * volume_dim.y * volume_dim.z))
	{ 
		// Get the x, y, z position of the voxel.
		vox_coordinate.z = threadIdxInGrid / (volume_dim.x * volume_dim.y);
		vox_coordinate.y = (threadIdxInGrid - (vox_coordinate.z * volume_dim.x * volume_dim.y)) / volume_dim.x;
		vox_coordinate.x = threadIdxInGrid - vox_coordinate.z * volume_dim.x * volume_dim.y - (vox_coordinate.y * volume_dim.x);

		// Calculate the distance of the voxel from the origin (in mm) along the x, y and z axes.
		distance_from_image_origin.x = img_origin.x + (pix_spacing.x * vox_coordinate.x);
		distance_from_image_origin.y = img_origin.y + (pix_spacing.y * vox_coordinate.y);
		distance_from_image_origin.z = img_origin.z + (pix_spacing.z * vox_coordinate.z);
		
		// Calculate the displacement of the voxel (in mm) in the x, y, and z directions.
		displacement_in_mm.x = distance_from_image_origin.x + tex1Dfetch(tex_dx, threadIdxInGrid); //dx[threadIdxInGrid];
		displacement_in_mm.y = distance_from_image_origin.y + tex1Dfetch(tex_dy, threadIdxInGrid); //dy[threadIdxInGrid];
		displacement_in_mm.z = distance_from_image_origin.z + tex1Dfetch(tex_dz, threadIdxInGrid); //dz[threadIdxInGrid];

		// Calculate the displacement value in terms of voxels.
		displacement_in_vox.x = (displacement_in_mm.x - img_offset.x) / pix_spacing.x;
		displacement_in_vox.y = (displacement_in_mm.y - img_offset.y) / pix_spacing.y;
		displacement_in_vox.z = (displacement_in_mm.z - img_offset.z) / pix_spacing.z;

		/*
		// Get the tile location of the voxel.
		p.x = vox_coordinate.x / vox_per_rgn.x;
		p.y = vox_coordinate.y / vox_per_rgn.y;
		p.z = vox_coordinate.z / vox_per_rgn.z;
				
		// Get the offset of the voxel within the tile.
		q.x = vox_coordinate.x - p.x * vox_per_rgn.x;
		q.y = vox_coordinate.y - p.y * vox_per_rgn.y;
		q.z = vox_coordinate.z - p.z * vox_per_rgn.z;
				
		// Use the tile location of the voxel to compute the index into the c_lut.
		pidx = ((p.z * rdims.y + p.y) * rdims.x) + p.x;
		prow = &c_lut[pidx*64];

		// Use the offset if the voxel to compute the index into the multiplier LUT or q_lut.
		qidx = ((q.z * vox_per_rgn.y + q.y) * vox_per_rgn.x) + q.x;
		qrow = &q_lut[qidx*64];
		*/

		// Check if the displaced voxel lies outside the region of interest.
		if ((displacement_in_vox.x < -0.5) || (displacement_in_vox.x > (volume_dim.x - 0.5)) || 
			(displacement_in_vox.y < -0.5) || (displacement_in_vox.y > (volume_dim.y - 0.5)) || 
			(displacement_in_vox.z < -0.5) || (displacement_in_vox.z > (volume_dim.z - 0.5))) {
			dc_dv_x[threadIdxInGrid] = 0.0;
			dc_dv_y[threadIdxInGrid] = 0.0;
			dc_dv_z[threadIdxInGrid] = 0.0;
		}
		else {

			// Clamp and interpolate along the X axis.
			displacement_in_vox_floor.x = floor(displacement_in_vox.x);
			displacement_in_vox_round.x = round(displacement_in_vox.x);
			if(displacement_in_vox_floor.x < 0){
				displacement_in_vox_floor.x = 0;
				displacement_in_vox_round.x = 0;
			}
			else if(displacement_in_vox_floor.x >= (volume_dim.x - 1)){
				displacement_in_vox_floor.x = volume_dim.x - 2;
				displacement_in_vox_round.x = volume_dim.x - 1;
			}
			
			// Clamp and interpolate along the Y axis.
			displacement_in_vox_floor.y = floor(displacement_in_vox.y);
			displacement_in_vox_round.y = round(displacement_in_vox.y);
			if(displacement_in_vox_floor.y < 0){
				displacement_in_vox_floor.y = 0;
				displacement_in_vox_round.y = 0;
			}
			else if(displacement_in_vox_floor.y >= (volume_dim.y - 1)){
				displacement_in_vox_floor.y = volume_dim.y - 2;
				displacement_in_vox_round.y = volume_dim.y - 1;
			}
			
			// Clamp and intepolate along the Z axis.
			displacement_in_vox_floor.z = floor(displacement_in_vox.z);
			displacement_in_vox_round.z = round(displacement_in_vox.z);
			if(displacement_in_vox_floor.z < 0){
				displacement_in_vox_floor.z = 0;
				displacement_in_vox_round.z = 0;
			}
			else if(displacement_in_vox_floor.z >= (volume_dim.z - 1)){
				displacement_in_vox_floor.z = volume_dim.z - 2;
				displacement_in_vox_round.z = volume_dim.z - 1;
			}

			// Compute spatial gradient using nearest neighbors.
			mvr = (((displacement_in_vox_round.z * volume_dim.y) + displacement_in_vox_round.y) * volume_dim.x) + displacement_in_vox_round.x;
			dc_dv_x[threadIdxInGrid] = diff[threadIdxInGrid] * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 0); //moving_grad[(3 * (int)mvr) + 0];
			dc_dv_y[threadIdxInGrid] = diff[threadIdxInGrid] * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 1); //moving_grad[(3 * (int)mvr) + 1];
			dc_dv_z[threadIdxInGrid] = diff[threadIdxInGrid] * tex1Dfetch(tex_moving_grad, (3 * (int)mvr) + 2); //moving_grad[(3 * (int)mvr) + 2];
			
			/*
		    for (int i = 0; i < 64; i++) {
				cidx = 3 * prow[i];
				grad[cidx+0] += dc_dv.x * qrow[i];
				grad[cidx+1] += dc_dv.y * qrow[i];
				grad[cidx+2] += dc_dv.z * qrow[i];
			}
			*/
		}
	}
}


/***********************************************************************
 * bspline_cuda_compute_score_kernel
 *
 * This kernel reduces the score stream to a single value.  It will work
 * for an aribtrary stream size, and also checks a flag for each element
 * to determine whether or not it is "valid" before adding it to the
 * final sum.
 ***********************************************************************/
__global__ void bspline_cuda_compute_score_kernel(
  float *idata, 
  float *odata, 
  int   *valid_voxels, 
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
  if(threadIdxInGrid >= num_elems || valid_voxels[threadIdxInGrid] == 0)
    sdata[threadIdxInBlock] = 0.0;
  else 
    sdata[threadIdxInBlock] = idata[threadIdxInGrid] * idata[threadIdxInGrid];

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
 * bspline_cuda_compute_grad_mean_kernel
 *
 * This kernel computes the value of grad_mean from the gradient stream.
 ***********************************************************************/
__global__ void bspline_cuda_compute_grad_mean_kernel(
	float *idata,
	float *odata,
	int num_elems)
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
	for(unsigned int s = (threadsPerBlock / 2); s > 0; s >>= 1) {
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
 * bspline_cuda_compute_grad_norm_kernel
 *
 * This kernel computes the value of grad_norm from the gradient stream.
 ***********************************************************************/
__global__ void bspline_cuda_compute_grad_norm_kernel(
	float *idata,
	float *odata,
	int num_elems)
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
		sdata[threadIdxInBlock] = fabs(idata[threadIdxInGrid]);

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


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_initialize_j_orig()
// 
// Initialize the GPU to execute bspline_cuda_score_j_mse().
// This is the origional version.  No zero copy of fanciness
//
// AUTHOR: James Shackleford
// DATE  : September 17, 2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_cuda_initialize_j_orig (
    Dev_Pointers_Bspline* dev_ptrs,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Bspline_xform* bxf,
    Bspline_parms* parms)
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


#if defined (commentout)
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
#endif


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
    Bspline_parms* parms)
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
// FUNCTION: bspline_cuda_initialize_h()
// 
// Initialize the GPU to execute bspline_cuda_score_h_mse().
//
// AUTHOR: James Shackleford
// DATE  : September 11, 2009
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_initialize_h(Dev_Pointers_Bspline* dev_ptrs,
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
//	int pad = 32 - (vox_per_tile % 32);

	dev_ptrs->LUT_Offsets_size = num_tiles*sizeof(int);

	cudaMalloc((void**)&dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);
	checkCUDAError("cudaMalloc(): dev_ptrs->LUT_Offsets");
	printf(".");

	cudaMemcpy(dev_ptrs->LUT_Offsets, offsets, dev_ptrs->LUT_Offsets_size, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy(): offsets --> dev_ptrs->LUT_Offsets");
//	cudaBindTexture(0, tex_LUT_Offsets, dev_ptrs->LUT_Offsets, dev_ptrs->LUT_Offsets_size);

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
#if defined (commentout)
    printf("Initializing CUDA (g) ... ");
#endif

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
    dc_dv_mem_size = 3 
	    * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2]
	    * bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float);
#if defined (commentout)
    printf ("vox_per_rgn (%d,%d,%d), rdim (%d,%d,%d), bytes %d\n", 
	    bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2],
	    bxf->rdims[0], bxf->rdims[1], bxf->rdims[2],
	    dc_dv_mem_size);
#endif
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

#if defined (commentout)
    printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
#endif
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
#if defined (commentout)
	printf("Initializing CUDA... ");
#endif

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

	// Allocate memory to hold the gradient values.
	if(cudaMalloc((void**)&gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad stream on GPU");
	if(cudaMalloc((void**)&gpu_grad_temp, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to allocate memory for the grad_temp stream on GPU");
	if(cudaBindTexture(0, tex_grad, gpu_grad, coeff_mem_size) != cudaSuccess)
		checkCUDAError("Failed to bind tex_grad to linear memory");
	total_bytes += 2 * coeff_mem_size;

#if defined (commentout)
	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
#endif
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

	printf("Total Memory Allocated on GPU: %d MB\n", total_bytes / (1024 * 1024));
}


/***********************************************************************
 * bspline_cuda_initialize_c
 * 
 * cuda "c" requires calling initialize_d after initialize_c.
 * 
 ***********************************************************************/
void bspline_cuda_initialize_c(
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
void bspline_cuda_copy_grad_to_host (float* host_grad)
{
    if (cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");
}

/***********************************************************************
 * bspline_cuda_calculate_run_kernels_g
 *
 * This function runs the kernels to calculate the score and gradient
 * as part of bspline_cuda_score_g_mse.
 ***********************************************************************/
void 
bspline_cuda_calculate_run_kernels_g (
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad,
    BSPLINE_Xform *bxf,
    BSPLINE_Parms *parms,
    int run_low_mem_version, 
    int debug)
{
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

    // Configure the grid.
    int threads_per_block;
    int num_threads;
    int num_blocks;
    int smemSize;

#if defined (commentout)
    if (debug) {
	sprintf (debug_fn, "dump_mse.txt");
	fp = fopen (debug_fn, "w");
    }
#endif

    if (!run_low_mem_version) {
	//printf("Launching one-shot version of bspline_cuda_score_g_mse_kernel1...\n");
		

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
//		printf("\nExecuting bspline_cuda_score_g_mse_kernel1() with Grid [%i,%i]...\n", Grid_x, Grid_y);
	}

	dim3 dimGrid1(Grid_x, Grid_y, 1);


//	dim3 dimGrid1(num_blocks / 128, 128, 1);
	dim3 dimBlock1(threads_per_block, 1, 1);
	// ----------------------------------------------------------

/*
  threads_per_block = 256;
  //threads_per_block = 64;
  num_threads = fixed->npix;
  num_blocks = (int)ceil(num_threads / (float)threads_per_block);
  dim3 dimGrid1(num_blocks / 128, 128, 1);
  dim3 dimBlock1(threads_per_block, 1, 1);
*/
	smemSize = 12 * sizeof(float) * threads_per_block;

	bspline_cuda_score_g_mse_kernel1<<<dimGrid1, dimBlock1, smemSize>>>
	    (
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

#if defined (commentout)
	if (debug) {
	    int ri, rj, rk;
	    int fi, fj, fk;
	    float *tmp = (float*) malloc (dc_dv_mem_size);
	    if (cudaMemcpy (tmp, gpu_dc_dv, dc_dv_mem_size,
			    cudaMemcpyDeviceToHost) != cudaSuccess) {
		checkCUDAError("Failed to copy gpu_dc_dv to CPU");
	    }

	    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
		for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
		    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
			int idx = 3 * (((rk * bxf->roi_dim[1]) + rj) * bxf->roi_dim[0] + ri);
			fprintf (fp, "%d %d %d %g %g %g\n", ri, rj, rk, 
				 tmp[idx+0], tmp[idx+1], tmp[idx+2]);
		    }
		}
	    }
	    free (tmp);
	}
#endif

    } else {
	int tiles_per_launch = 512;
	//printf("Launching low memory version of bspline_cuda_score_g_mse_kernel1 with %d tiles per launch. \n", tiles_per_launch);
		
	threads_per_block = 256;
	num_threads = tiles_per_launch * vox_per_rgn.x * vox_per_rgn.y * vox_per_rgn.z;
	num_blocks = (int)ceil(num_threads / (float)threads_per_block);
	dim3 dimGrid1(num_blocks / 128, 128, 1);
	dim3 dimBlock1(threads_per_block, 1, 1);
	smemSize = 12 * sizeof(float) * threads_per_block;

	for (int i = 0; i < rdims.x * rdims.y * rdims.z; i += tiles_per_launch) {
	    bspline_cuda_score_g_mse_kernel1_low_mem<<<dimGrid1, dimBlock1, smemSize>>>
		(
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

#if defined (commentout)
    if (debug) {
	fclose (fp);
    }
#endif


    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_g_mse_compute_score failed");

    // Reconfigure the grid.
    threads_per_block = 256;
    //    threads_per_block = 64;
    num_threads = bxf->num_knots;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
    smemSize = 15 * sizeof(float) * threads_per_block;

    //printf("Launching bspline_cuda_score_f_mse_kernel2...");
    bspline_cuda_score_g_mse_kernel2<<<dimGrid2, dimBlock2, smemSize>>>
	(
	    gpu_dc_dv,
	    gpu_grad,
	    num_threads,
	    rdims,
	    cdims,
	    vox_per_rgn);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("\bspline_cuda_score_g_mse_kernel2 failed");
}

/***********************************************************************
 * bspline_cuda_calculate_run_kernels_f
 *
 * This function runs the kernels to calculate the score and gradient
 * as part of bspline_cuda_score_f_mse.
 ***********************************************************************/
void
bspline_cuda_calculate_run_kernels_f
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms)
{
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

    /*
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
    */

	
    // Configure the grid.
    int threads_per_block = 256;
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

    threads_per_block = 256;
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
	
    // Reconfigure the grid.
    threads_per_block = 256;
    num_threads = bxf->num_knots;
    num_blocks = (int)ceil(num_threads / (float)threads_per_block);
    dim3 dimGrid2(num_blocks, 1, 1);
    dim3 dimBlock2(threads_per_block, 1, 1);
    int smemSize = 15 * sizeof(float) * threads_per_block;

    //printf("Launching bspline_cuda_score_f_mse_kernel2...");
	
    bspline_cuda_score_f_mse_kernel2_nk<<<dimGrid2, dimBlock2, smemSize>>>(
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
}

/***********************************************************************
 * bspline_cuda_final_steps_f
 *
 * This function performs sum reduction of the score and gradient
 * streams as part of bspline_cuda_score_f_mse.
 ***********************************************************************/
void 
bspline_cuda_final_steps_f
(
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
	int Grid_x = 0;
	int Grid_y = 0;
	int num_elems = volume_dim[0] * volume_dim[1] * volume_dim[2];
//    int num_blocks = (int)ceil(num_elems / 512.0);

	// ---
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
	// ---

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

    // Calculate grad_norm and grad_mean.

    // Reconfigure the grid.
	Grid_x = 0;
	Grid_y = 0;
	int num_vox = fixed->dim[0] * fixed->dim[1] * fixed->dim[2];
	num_elems = bxf->num_coeff;
//    num_blocks = (int)ceil(num_elems / 512.0);

	// ---
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
		printf("\n[ERROR] Unable to find suitable gradient final steps configuration!\n");
		exit(0);
	} else {
//		printf("\nExecuting sum_reduction_kernel() with Grid [%i,%i]...\n", Grid_x, Grid_y);
	}

	dim3 dimGrid2(Grid_x, Grid_y, 1);
	// ---


    dim3 dimBlock2(128, 2, 2);
    smemSize = 512 * sizeof(float);

    //    printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid2, dimBlock2>>>(
							     gpu_grad,
							     num_vox,
							     num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");

    if(cudaMemcpy(host_grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");

#if defined (commentout)
    /* kkk */
    printf ("host_grad[0] = %g\n", host_grad[0]);
    printf ("host_grad[5] = %g\n", host_grad[5]);
    exit (0);
#endif

    // printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
    bspline_cuda_compute_grad_mean_kernel<<<dimGrid2, dimBlock2, smemSize>>>
	    (
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
    bspline_cuda_compute_grad_norm_kernel<<<dimGrid2, dimBlock2, smemSize>>>
	    (
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
void bspline_cuda_calculate_score_e
(
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
void 
bspline_cuda_run_kernels_e_v2
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms,
 int sidx0,
 int sidx1,
 int sidx2)
{
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
	
    // Clear the dc_dv values.
    bspline_cuda_clear_dc_dv();

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
}

/***********************************************************************
 * bspline_cuda_run_kernels_e
 *
 * This function runs the kernels to compute both the score and dc_dv
 * values for a given set as part of bspline_cuda_score_e_mse.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_e
(
 Volume *fixed,
 Volume *moving,
 Volume *moving_grad,
 BSPLINE_Xform *bxf,
 BSPLINE_Parms *parms,
 int sidx0,
 int sidx1,
 int sidx2)
{
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
	
    // Clear the dc_dv values.
    bspline_cuda_clear_dc_dv();

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
void 
bspline_cuda_final_steps_e_v2
(
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
    // Calculate the set dimensions.
#if defined (commentout)
    int3 sdims;
    sdims.x = (int)ceil(bxf->rdims[0] / 4.0);
    sdims.y = (int)ceil(bxf->rdims[1] / 4.0);
    sdims.z = (int)ceil(bxf->rdims[2] / 4.0);
#endif

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
void bspline_cuda_final_steps_e
(
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
}

/***********************************************************************
 * bspline_cuda_run_kernels_d
 *
 * This function runs the kernels to compute the score and dc_dv values
 * for a given tile as part of bspline_cuda_score_d_mse.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_d
(
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
}

/***********************************************************************
 * bspline_cuda_final_steps_e
 *
 * This function runs the kernels necessary to reduce the score and
 * gradient streams to a single value as part of bspline_cuda_score_d_mse.
 ***********************************************************************/
void 
bspline_cuda_final_steps_d
(
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
 * bspline_cuda_run_kernels_c
 *
 * This function runs the kernels necessary to compute the score and
 * dc_dv values as part of bspline_cuda_score_c_mse.
 ***********************************************************************/
void 
bspline_cuda_run_kernels_c
(
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

    //printf("Launching bspline_cuda_compute_dxyz_kernel... ");
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

    //printf("Launching bspline_cuda_compute_diff_kernel... ");
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

    if(cudaMemcpy(host_diff, gpu_diff, fixed->npix * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy diff stream from GPU to host");

    //printf("Launching bspline_cuda_compute_dc_dv_kernel... ");
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

    //printf("Launching bspline_cuda_compute_score_kernel... ");
    bspline_cuda_compute_score_kernel<<<dimGrid, dimBlock, smemSize>>>(
	gpu_diff,
	gpu_diff,
	gpu_valid_voxels,
	num_elems
    );

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_compute_score_kernel failed");

    sum_reduction_last_step_kernel<<<dimGrid, dimBlock>>>(
	gpu_diff,
	gpu_diff,
	num_elems
    );

    cudaThreadSynchronize();

    if(cudaMemcpy(host_score, gpu_diff,  sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy score from GPU to host");

    *host_score = *host_score / num_elems;

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
void 
bspline_cuda_calculate_gradient_c
(
 BSPLINE_Parms* parms, 
 Bspline_state* bst,
 BSPLINE_Xform* bxf,
 Volume *fixed,
 float *host_grad_norm,
 float *host_grad_mean) 
{
    BSPLINE_Score* ssd = &bst->ssd;
	
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

    //printf("Launching bspline_cuda_update_grad_kernel... ");
    bspline_cuda_update_grad_kernel<<<dimGrid, dimBlock>>>(
							   gpu_grad,
							   num_vox,
							   num_elems);

    if(cudaThreadSynchronize() != cudaSuccess)
	checkCUDAError("bspline_cuda_update_grad_kernel failed");

    if(cudaMemcpy(ssd->grad, gpu_grad, coeff_mem_size, cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy gpu_grad to CPU");

    //printf("Launching bspline_cuda_compute_grad_mean_kernel... ");
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

    if(cudaMemcpy(host_grad_mean, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_mean from GPU to host");

    //printf("Launching bspline_cuda_compute_grad_norm_kernel... ");
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

    if(cudaMemcpy(host_grad_norm, gpu_grad_temp, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	checkCUDAError("Failed to copy grad_norm from GPU to host");
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



/***********************************************************************
 * bspline_cuda_clean_up_g
 *
 * This function frees all allocated memory on the GPU for version "g".
 ***********************************************************************/
void bspline_cuda_clean_up_g() 
{
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
}

/***********************************************************************
 * bspline_cuda_clean_up_f
 *
 * This function frees all allocated memory on the GPU for version "f".
 ***********************************************************************/
void bspline_cuda_clean_up_f() 
{
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
void bspline_cuda_clean_up_d() 
{
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
 * bspline_cuda_clean_up_c
 *
 * This function frees all allocated memory on the GPU for version "c".
 ***********************************************************************/
void bspline_cuda_clean_up_c() 
{
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
