/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

/****************************************************\
* Uncomment the line below to enable verbose output. *
* Enabling this should not nerf performance.         *
\****************************************************/
#define VERBOSE 1

/**********************************************************\
* Uncomment the line below to enable detailed performance  *
* reporting.  This measurement alters the system, however, *
* resulting in significantly slower kernel execution.      *
\**********************************************************/
#define TIME_KERNEL
#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "cuda_utils.h"
#include "drr_cuda.h"
#include "drr_cuda_p.h"
#include "drr_opts.h"
#include "file_util.h"
#include "math_util.h"
#include "proj_image.h"
#include "ray_trace_exact.h"
#include "volume.h"
#include "timer.h"

// P R O T O T Y P E S ////////////////////////////////////////////////////
__global__ void kernel_drr (float * dev_vol,  int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing);


// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
texture<float, 1, cudaReadModeElementType> tex_coef;
texture<float, 3, cudaReadModeElementType> tex_3Dvol;

// uses 3D textures and pre-calculated coefs to accelerate DRR generation.
void kernel_drr (
    float * dev_img, 
    int2 img_dim, 
    float2 ic, 
    float3 nrm, 
    float sad, 
    float scale, 
    float3 vol_offset, 
    int3 vol_dim, 
    float3 vol_pix_spacing
)
{
    extern __shared__ float sdata[];
    float3 vp;
    int i,j,k;
    int x,y,xy7;
    float vol;

    unsigned int tid = threadIdx.x;

    /* Get coordinates of this image pixel */
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Compute ray */
#if defined (commentout)
    vec3_copy (r_tgt, ul_room);
    vec3_scale3 (tmp, incr_r, (double) r);
    vec3_add2 (r_tgt, tmp);
    int idx = c - options->image_window[2] 
	+ (r - options->image_window[0]) * cols;
    vec3_scale3 (tmp, incr_c, (double) c);
    vec3_add3 (p2, r_tgt, tmp);
#endif
    /* (TBD) */

    /* Loop through ray */

    /* Write output pixel value */
    dev_img[y*img_dim.x + x] = x;
}

void*
drr_cuda_state_create (
    Proj_image *proj,
    Volume *vol,
    Drr_options *options
)
{
    Drr_cuda_state *state;
    Drr_kernel_args *kargs;

    state = (Drr_cuda_state *) malloc (sizeof(Drr_cuda_state));
    memset (state, 0, sizeof(Drr_cuda_state));

    state->kargs = kargs = (Drr_kernel_args*) malloc (sizeof(Drr_kernel_args));
    cudaMalloc ((void**) &state->dev_matrix, 12 * sizeof(float));
    cudaMalloc ((void**) &state->dev_kargs, sizeof(Drr_kernel_args));

    printf ("printf state = %p\n", state);
    printf ("printf state->kargs = %p\n", state->kargs);

    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];

    // prepare texture
    cudaChannelFormatDesc ca_descriptor;
    cudaExtent ca_extent;
    cudaArray *dev_3Dvol=0;

    ca_descriptor = cudaCreateChannelDesc<float>();
    ca_extent.width  = vol->dim[0];
    ca_extent.height = vol->dim[1];
    ca_extent.depth  = vol->dim[2];
    cudaMalloc3DArray (&dev_3Dvol, &ca_descriptor, ca_extent);
    cudaBindTextureToArray (tex_3Dvol, dev_3Dvol, ca_descriptor);

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent   = ca_extent;
    cpy_params.kind     = cudaMemcpyHostToDevice;
    cpy_params.dstArray = dev_3Dvol;

    //http://sites.google.com/site/cudaiap2009/cookbook-1#TOC-CUDA-3D-Texture-Example-Gerald-Dall
    // The pitched pointer is really tricky to get right. We give the
    // pitch of a row, then the number of elements in a row, then the
    // height, and we omit the 3rd dimension.
    cpy_params.srcPtr = make_cudaPitchedPtr ((void*)vol->img, 
	ca_extent.width * sizeof(float), ca_extent.width , ca_extent.height);

    cudaMemcpy3D (&cpy_params);

    cudaMalloc ((void**) &state->dev_img, 
	options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));

    cudaMalloc ((void**) &state->dev_coef, 
	7 * options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));
    cuda_utils_check_error ("Unable to allocate coef devmem");
    state->host_coef = (float*) malloc (
	7 * options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));
		
    return (void*) state;
}

void
drr_cuda_state_destroy (
    void *void_state
)
{
    Drr_cuda_state *state = (Drr_cuda_state*) void_state;
    
    cudaFree (state->dev_img);
    cudaFree (state->dev_kargs);
    cudaFree (state->dev_matrix);
    cudaFree (state->dev_coef);
    free (state->host_coef);
    free (state->kargs);
}

void
drr_cuda_ray_trace_image (
    Proj_image *proj, 
    Volume *vol, 
    Volume_limit *vol_limit, 
    double p1[3], 
    double ul_room[3], 
    double incr_r[3], 
    double incr_c[3], 
    void *dev_state, 
    Drr_options *options
)
{
    Timer timer, total_timer;
    double time_kernel = 0;
    int i;

    // CUDA device pointers
    Drr_cuda_state *state = (Drr_cuda_state*) dev_state;
    Drr_kernel_args *kargs = state->kargs;

    // Start the timer
    plm_timer_start (&total_timer);
    plm_timer_start (&timer);

    // Load dynamic kernel arguments (different for each projection)
    kargs->img_dim.x = proj->dim[0];
    kargs->img_dim.y = proj->dim[1];
    kargs->ic.x = proj->pmat->ic[0];
    kargs->ic.y = proj->pmat->ic[1];
    kargs->nrm.x = proj->pmat->nrm[0];
    kargs->nrm.y = proj->pmat->nrm[1];
    kargs->nrm.z = proj->pmat->nrm[2];
    kargs->sad = proj->pmat->sad;
    kargs->sid = proj->pmat->sid;
    for (i = 0; i < 12; i++) {
	kargs->matrix[i] = (float) proj->pmat->matrix[i];
    }
    kargs->p1.x = p1[0];
    kargs->p1.y = p1[1];
    kargs->p1.z = p1[2];
    kargs->ul_room.x = ul_room[0];
    kargs->ul_room.y = ul_room[1];
    kargs->ul_room.z = ul_room[2];

    cudaMemcpy (state->dev_matrix, kargs->matrix, sizeof(kargs->matrix), 
	cudaMemcpyHostToDevice);
    cudaBindTexture (0, tex_matrix, state->dev_matrix, sizeof(kargs->matrix));

    // Thread Block Dimensions
    int tBlock_x = 16;
    int tBlock_y = 16;

    // Each element in the image gets 1 thread
    int blocksInX = (vol->dim[0]+tBlock_x-1)/tBlock_x;
    int blocksInY = (vol->dim[1]+tBlock_y-1)/tBlock_y;
    dim3 dimGrid  = dim3(blocksInX, blocksInY);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y);

    // Note: proj->img AND proj->matrix are passed via texture memory

    //int smemSize = vol->dim[0]  * sizeof(float);

    printf ("Preprocessing time: %f secs\n", plm_timer_report (&timer));
    plm_timer_start (&timer);

    // Invoke ze kernel  \(^_^)/
    kernel_drr<<< dimGrid, dimBlock>>> (
	state->dev_img, 
	kargs->img_dim,
	kargs->ic,
	kargs->nrm,
	kargs->sad,
	kargs->scale,
	kargs->vol_offset,
	kargs->vol_dim,
	kargs->vol_pix_spacing);

    printf ("Kernel time: %f secs\n", plm_timer_report (&timer));
    plm_timer_start (&timer);

    cuda_utils_check_error("Kernel Panic!");

#if defined (TIME_KERNEL)
    // CUDA kernel calls are asynchronous...
    // In order to accurately time the kernel
    // execution time we need to set a thread
    // barrier here after its execution.
    cudaThreadSynchronize();
#endif

    time_kernel += plm_timer_report (&timer);

    // Unbind the image and projection matrix textures
    //cudaUnbindTexture( tex_img );
    cudaUnbindTexture (tex_matrix);
    cudaUnbindTexture (tex_coef);

    // Copy reconstructed volume from device to host
    //cudaMemcpy( vol->img, dev_vol, vol->npix * vol->pix_size, cudaMemcpyDeviceToHost );
    cudaMemcpy (proj->img, state->dev_img, 
	proj->dim[0] * proj->dim[1] * sizeof(float), 
	cudaMemcpyDeviceToHost);
    cuda_utils_check_error("Error: Unable to retrieve data volume.");
}
