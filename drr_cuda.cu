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

#include "drr_cuda.h"
#include "drr_cuda_p.h"
#include "drr_opts.h"
#include "file_util.h"
#include "mathutil.h"
#include "proj_image.h"
#include "volume.h"
#include "timer.h"


// P R O T O T Y P E S ////////////////////////////////////////////////////
void checkCUDAError(const char *msg);

__global__ void kernel_drr_i (float * dev_img,  float * dev_vol, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing);

__global__ void kernel_drr_i3 (float * dev_vol,  int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing);

__global__ void kernel_drr_j (float * dev_img,  float * dev_vol, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing);

///////////////////////////////////////////////////////////////////////////



// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
texture<float, 1, cudaReadModeElementType> tex_coef;
texture<float, 3, cudaReadModeElementType> tex_3Dvol;
///////////////////////////////////////////////////////////////////////////



//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
__global__
void kernel_drr_i (float * dev_vol,  float * dev_img, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing)
{
    // CUDA 2.0 does not allow for a 3D grid, which severely
    // limits the manipulation of large 3D arrays of data.  The
    // following code is a hack to bypass this implementation
    // limitation.
    extern __shared__ float sdata[];
    float3 ip;
    float3 vp;
    int i,j,k;
    long int vol_idx;

    unsigned int tid = threadIdx.x;

    ip.x = __int2float_rn(blockIdx.x)-ic.x;
    ip.y = __int2float_rn(blockIdx.y)-ic.y;
	
    vp.x=vol_offset.x+threadIdx.x*vol_pix_spacing.x;

    vp.y=(ip.y*tex1Dfetch(tex_matrix, 8)-tex1Dfetch(tex_matrix, 4))*vp.x+ip.y*tex1Dfetch(tex_matrix, 11);

    vp.y/=tex1Dfetch(tex_matrix, 5)-ip.y*tex1Dfetch(tex_matrix, 9);

    vp.z=ip.x*(tex1Dfetch(tex_matrix, 8)*vp.x+tex1Dfetch(tex_matrix, 9)*vp.y+tex1Dfetch(tex_matrix, 11));

    vp.z/=tex1Dfetch(tex_matrix, 2);

    i=  threadIdx.x;

    j=  __float2int_rd((vp.y-vol_offset.y)/vol_pix_spacing.y);

    k=  __float2int_rd((vp.z-vol_offset.z)/vol_pix_spacing.z);

    if (j<0||j>=vol_dim.y||k<0||k>=vol_dim.z)

	sdata[tid]=0.0f;
    else{
	vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );
	sdata[tid]=(dev_vol[vol_idx]+1000.0f);
    }

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        sdata[tid] += sdata[tid + 32]; EMUSYNC;
        sdata[tid] += sdata[tid + 16]; EMUSYNC;
        sdata[tid] += sdata[tid +  8]; EMUSYNC;
        sdata[tid] += sdata[tid +  4]; EMUSYNC;
        sdata[tid] += sdata[tid +  2]; EMUSYNC;
        sdata[tid] += sdata[tid +  1]; EMUSYNC;
    }

    // do reduction in shared mem





    // write result for this block to global mem
    if (tid == 0) 
	dev_img[blockIdx.x*img_dim.y + blockIdx.y] = sdata[0];


}

void kernel_drr_j (float * dev_vol,  float * dev_img, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing)
{
	// CUDA 2.0 does not allow for a 3D grid, which severely
	// limits the manipulation of large 3D arrays of data.  The
	// following code is a hack to bypass this implementation
	// limitation.
	extern __shared__ float sdata[];
	float3 ip;
	float3 vp;
	int i,j,k;
	long int vol_idx;

	unsigned int tid = threadIdx.x;

	ip.y = __int2float_rn(blockIdx.y)-ic.y;
	ip.x = __int2float_rn(blockIdx.x)-ic.x;
	
	vp.y=vol_offset.y+threadIdx.x*vol_pix_spacing.y;

	vp.x=(ip.y*tex1Dfetch(tex_matrix, 9)-tex1Dfetch(tex_matrix, 5))*vp.y+ip.y*tex1Dfetch(tex_matrix, 11);

	vp.x/=tex1Dfetch(tex_matrix, 4)-ip.y*tex1Dfetch(tex_matrix, 8);

	vp.z=ip.x*(tex1Dfetch(tex_matrix, 8)*vp.x+tex1Dfetch(tex_matrix, 9)*vp.y+tex1Dfetch(tex_matrix, 11));

	vp.z/=tex1Dfetch(tex_matrix, 2);

	i=  __float2int_rd((vp.x-vol_offset.x)/vol_pix_spacing.x);

	j=  threadIdx.x;

	k=  __float2int_rd((vp.z-vol_offset.z)/vol_pix_spacing.z);

	if (i<0||i>=vol_dim.x||k<0||k>=vol_dim.z)

		sdata[tid]=0.0f;
	else{
		//wrong?
		 vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );
		sdata[tid]=(dev_vol[vol_idx]+1000.0f);
		if (sdata[tid]<0)	sdata[tid]=0;
		if (sdata[tid]>2000) sdata[tid]=2000;
	}

	__syncthreads();

	  // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        sdata[tid] += sdata[tid + 32]; EMUSYNC;
        sdata[tid] += sdata[tid + 16]; EMUSYNC;
        sdata[tid] += sdata[tid +  8]; EMUSYNC;
        sdata[tid] += sdata[tid +  4]; EMUSYNC;
        sdata[tid] += sdata[tid +  2]; EMUSYNC;
        sdata[tid] += sdata[tid +  1]; EMUSYNC;
    }

  //  // do reduction in shared mem
  //  for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
  //      if (tid < s) {
  //          sdata[tid] += sdata[tid + s];
  //      }
  //      __syncthreads();
  //  }

  //  // write result for this block to global mem
  //  if (tid == 0) 
		//dev_img[blockIdx.x*img_dim.y + blockIdx.y] = sdata[0];


}


//DRR3  uses 3D textures and pre-calculated coefs to accelerate DRR generation.

void kernel_drr_i3 (float * dev_img, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing)
{
    // CUDA 2.0 does not allow for a 3D grid, which severely
    // limits the manipulation of large 3D arrays of data.  The
    // following code is a hack to bypass this implementation
    // limitation.
    extern __shared__ float sdata[];
    float3 vp;
    int i,j,k;
    int x,y,xy7;
    float vol;

    unsigned int tid = threadIdx.x;

    x = blockIdx.x;
    y = blockIdx.y;
    xy7=7*(y*img_dim.x+x);
	
    if (abs(tex1Dfetch(tex_matrix, 5))>abs(tex1Dfetch(tex_matrix, 4))) {
	vp.x=vol_offset.x+threadIdx.x*vol_pix_spacing.x;
	vp.y=tex1Dfetch(tex_coef, xy7)*vp.x+tex1Dfetch(tex_coef, xy7+1);
	vp.z=tex1Dfetch(tex_coef, xy7+4)*vp.x
	    +tex1Dfetch(tex_coef, xy7+5)*vp.y+tex1Dfetch(tex_coef, xy7+6);

	i=  threadIdx.x;
	j=  __float2int_rd((vp.y-vol_offset.y)/vol_pix_spacing.y);
	k=  __float2int_rd((vp.z-vol_offset.z)/vol_pix_spacing.z);

	//if (j<0||j>=vol_dim.y||k<0||k>=vol_dim.z)
	if ((i-vol_dim.x/2)*(i-vol_dim.x/2)+(j-vol_dim.y/2)*(j-vol_dim.y/2)
	    > vol_dim.y*vol_dim.y/4||k<0||k>=vol_dim.z) 
	{
	    sdata[tid]=0.0f;
	} else {
	    vol=tex3D(tex_3Dvol,i,j,k);
	    sdata[tid]=(vol+1000.0f);
	}
    } else {
	vp.y=vol_offset.y+threadIdx.x*vol_pix_spacing.y;
	vp.x=tex1Dfetch(tex_coef, xy7+2)*vp.y+tex1Dfetch(tex_coef, xy7+3);
	vp.z=tex1Dfetch(tex_coef, xy7+4)*vp.x
	    +tex1Dfetch(tex_coef, xy7+5)*vp.y+tex1Dfetch(tex_coef, xy7+6);
	j=  threadIdx.x;
	i=  __float2int_rd((vp.x-vol_offset.x)/vol_pix_spacing.x);
	k=  __float2int_rd((vp.z-vol_offset.z)/vol_pix_spacing.z);

	if ((i-vol_dim.x/2)*(i-vol_dim.x/2)+(j-vol_dim.y/2)*(j-vol_dim.y/2)
	    > vol_dim.y*vol_dim.y/4||k<0||k>=vol_dim.z)
	{
	    sdata[tid]=0.0f;
	} else {
	    vol=tex3D(tex_3Dvol,i,j,k);
	    sdata[tid]=(vol+1000.0f);
	}
    }

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        sdata[tid] += sdata[tid + 32]; EMUSYNC;
        sdata[tid] += sdata[tid + 16]; EMUSYNC;
        sdata[tid] += sdata[tid +  8]; EMUSYNC;
        sdata[tid] += sdata[tid +  4]; EMUSYNC;
        sdata[tid] += sdata[tid +  2]; EMUSYNC;
        sdata[tid] += sdata[tid +  1]; EMUSYNC;
    }

    // write result for this block to global mem
    if (tid == 0) 
	dev_img[blockIdx.x*img_dim.y + blockIdx.y] = sdata[0];
}



//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_



//////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_DRR() //////////////////////////////////
#if defined (commentout)
extern "C"
int CUDA_DRR (Volume *vol, Drr_options *options)
{
    Timer timer, total_timer;
    double time_kernel = 0;
    double time_io = 0;
    double time_total = 0;
    Drr_kernel_args *kargs;
    Proj_image* cbi;
    int image_num;
    int a, i;

    // CUDA device pointers
    float *dev_vol;	            // Holds voxels on device
    float *dev_img;	            // Holds image pixels on device
    float *dev_matrix;
    Drr_kernel_args *dev_kargs; // Holds kernel parameters

    // Start the timer
    plm_timer_start (&total_timer);

    cudaMalloc ((void**)&dev_matrix, 12*sizeof(float) );
    cudaMalloc ((void**)&dev_kargs, sizeof (Drr_kernel_args));
    kargs = (Drr_kernel_args *) malloc (sizeof(Drr_kernel_args));
    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];

    cudaMalloc( (void**)&dev_vol, vol->npix*sizeof(float));
    checkCUDAError("Unable to allocate data volume");
	
    cudaMalloc ((void**)&dev_img, 
	options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));

    for (a = 0; a < options->num_angles; a++)
    {
	plm_timer_start (&timer);

	/* Copied from drr_c.c */
	double vup[3] = {0, 0, 1};
	double tgt[3] = {0.0, 0.0, 0.0};
	double nrm[3];
	double tmp[3];
	double cam[3];
	cam[0] = cos(angle);
	cam[1] = sin(angle);
	cam[2] = 0.0;
	vec3_sub3 (nrm, tgt, cam);
	vec3_normalize1 (nrm);
	vec3_scale3 (tmp, nrm, sad);
	vec3_copy (cam, tgt);
	vec3_sub2 (cam, tmp);

	// Load dynamic kernel arguments
	kargs->img_dim.x = options->image_resolution[0];
	kargs->img_dim.y = options->image_resolution[1];
	kargs->ic.x = options->image_center[0];
	kargs->ic.y = options->image_center[1];
	kargs->nrm.x = nrm[0];
	kargs->nrm.y = nrm[1];
	kargs->nrm.z = nrm[2];
	kargs->sad = options->sad;
	kargs->sid = options->sid;
	for (i=0; i<12; i++) {
	    kargs->matrix[i] = (float) cbi->matrix[i];
	}

	cudaMemcpy (dev_vol, vol->img, vol->npix * vol->pix_size, 
	    cudaMemcpyHostToDevice);
	cudaMemcpy (dev_matrix, kargs->matrix, sizeof(kargs->matrix), 
	    cudaMemcpyHostToDevice );
	cudaBindTexture( 0, tex_matrix, dev_matrix, sizeof(kargs->matrix)); 

	time_io += plm_timer_report (&timer);
	plm_timer_start (&timer);

	// Thead Block Dimensions
	int tBlock_x = vol->dim[0];
	int tBlock_y = 1;
	int tBlock_z = 1;

	// Each element in the volume (each voxel) gets 1 thread
	int blocksInX = cbi->dim[0];
	int blocksInY = cbi->dim[1];
	dim3 dimGrid  = dim3(blocksInX, blocksInY);
	dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

	// Invoke ze kernel  \(^_^)/

	int smemSize = vol->dim[0]  * sizeof(float);
	if (abs(kargs->matrix[5])>abs(kargs->matrix[4]))
	    //-------------------------------------
	    kernel_drr_i<<< dimGrid, dimBlock,  smemSize>>>(dev_vol,dev_img,
		kargs->img_dim,
		kargs->ic,
		kargs->nrm,
		kargs->sad,
		kargs->scale,
		kargs->vol_offset,
		kargs->vol_dim,
		kargs->vol_pix_spacing);
	else
	    //-------------------------------------
	    kernel_drr_j<<< dimGrid, dimBlock,  smemSize>>>(dev_vol,dev_img,
		kargs->img_dim,
		kargs->ic,
		kargs->nrm,
		kargs->sad,
		kargs->scale,
		kargs->vol_offset,
		kargs->vol_dim,
		kargs->vol_pix_spacing);
	checkCUDAError("Kernel Panic!");
	printf (" %d\n", image_num);

#if defined (TIME_KERNEL)
	// CUDA kernel calls are asynchronous...
	// In order to accurately time the kernel
	// execution time we need to set a thread
	// barrier here after its execution.
	cudaThreadSynchronize();
#endif

	// Unbind the image and projection matrix textures
	//cudaUnbindTexture( tex_img );
	cudaUnbindTexture( tex_matrix );

	// Copy reconstructed volume from device to host
	cudaMemcpy (cbi->img, dev_img, cbi->dim[0] * cbi->dim[1] * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAError ("Error: Unable to retrieve data volume.");
		
	char img_file[1024];
	
	size_t rc;
	FILE* fp;
	//sprintf (fmt, "%s\\%s\\%s", options->input_dir,options->sub_dir,img_file_pat);
	//sprintf (fmt, "%s\\%s", options->input_dir,img_file_pat);
	//   sprintf (img_file, fmt, image_num);
	//   sprintf (fmt, "%s\\%s", options->input_dir, mat_file_pat);
	//   sprintf (mat_file, fmt, image_num);
	//   return load_and_filter_cb_image (options,img_file, mat_file);
	sprintf (img_file, "%s\\DRR\\Proj_%03d.raw", options->input_dir,image_num);
	//   sprintf (img_file, fmt, image_num);
	//   sprintf (fmt, "%s\\%s", options->input_dir, mat_file_pat);




	fp = fopen (img_file,"wb");
	if (!fp) {
	    fprintf (stderr, "Can't open file %s for write\n. Skipped", img_file);
	    return(1);
	}
	float writeimg[512*384];
	for (int i=0; i<512*384; i++)
	    writeimg[i]=65535*exp(-cbi->img[i]/30000);


	/* write pixels */
	rc = fwrite (writeimg , sizeof(float),  512* 384, fp); 
	if (rc != 512 * 384) {
	    fprintf (stderr, "Couldn't write raster data for %s\n",
		img_file);
	    return(1);
	}
	printf("Writing OK\n");
			
	fclose(fp);

	time_kernel += plm_timer_report (&timer);
    }

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif
	
    // Report Timing Data
    time_total = plm_timer_report (&total_timer);
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", time_total);
#if defined (TIME_KERNEL)
    printf ("\tTotal Kernel  Time: %.9fs\n", time_kernel);
    printf ("\tTotal File IO Time: %.9fs\n\n", time_io);
#endif

    printf ("[Average Projection Time: %.9fs ]\n", time_total / (1+(options->last_img - options->first_img) / options->skip_img));
#if defined (TIME_KERNEL)
    printf ("\tAverage Kernel  Time: %.9fs\n", time_kernel / (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage File IO Time: %.9fs\n\n", time_io / (1+(options->last_img - options->first_img) / options->skip_img));
#endif
    printf("========================================\n");

    // Cleanup
    cudaFree( dev_img );
    cudaFree( dev_kargs );
    cudaFree( dev_matrix );
    cudaFree( dev_vol );	

    return 0;
}
#endif

//DRR3 uses 3D textures and pre-calculated coefs to accelerate DRR generation.

//////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_DRR() //////////////////////////////////
int CUDA_DRR3 (Volume *vol, Drr_options *options)
{
    Timer timer, total_timer;
    double time_kernel = 0;
    double time_io = 0;
    double time_total = 0;
    Proj_image* cbi;
    int image_num;
    int a, i;

    // CUDA device pointers
    Drr_kernel_args *kargs;
    Drr_kernel_args *dev_kargs;     // Holds kernel parameters on device
    float *dev_img;	            // Holds image pixels on device
    float *dev_matrix;
    float *dev_coef;
    float *host_coef;

    // Start the timer
    plm_timer_start (&total_timer);

    kargs = (Drr_kernel_args*) malloc (sizeof(Drr_kernel_args));
    cudaMalloc ((void**)&dev_matrix, 12*sizeof(float));
    cudaMalloc ((void**)&dev_kargs, sizeof(Drr_kernel_args));

    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];

    //Create DRR directory
    char drr_dir[1024];
    //    sprintf (drr_dir, "%s/DRR", options->input_dir);
    //    make_directory (drr_dir);
    printf ("GCS: Warning, output prefix not yet handled in cuda...\n");

    // prepare texture
    cudaChannelFormatDesc ca_descriptor;
    cudaExtent ca_extent;
    cudaArray *dev_3Dvol=0;

    ca_descriptor = cudaCreateChannelDesc<float>();
    ca_extent.width  = vol->dim[0];
    ca_extent.height = vol->dim[1];
    ca_extent.depth  = vol->dim[2];
    cudaMalloc3DArray( &dev_3Dvol, &ca_descriptor, ca_extent );
    cudaBindTextureToArray( tex_3Dvol, dev_3Dvol, ca_descriptor );

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent   = ca_extent;
    cpy_params.kind     = cudaMemcpyHostToDevice;
    cpy_params.dstArray = dev_3Dvol;

    //http://sites.google.com/site/cudaiap2009/cookbook-1#TOC-CUDA-3D-Texture-Example-Gerald-Dall
    // The pitched pointer is really tricky to get right. We give the
    // pitch of a row, then the number of elements in a row, then the
    // height, and we omit the 3rd dimension.
    cpy_params.srcPtr = make_cudaPitchedPtr ((void*)vol->img, 
	ca_extent.width *sizeof(float), ca_extent.width , ca_extent.height);

    cudaMemcpy3D ( &cpy_params );

    // cudaMemcpy(dev_vol,  vol->img, vol->npix * vol->pix_size, cudaMemcpyHostToDevice );

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif

    cudaMalloc ((void**)&dev_img, 
	options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));

    cudaMalloc ((void**)&dev_coef, 
	7 * options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));
    checkCUDAError ("Unable to allocate coef devmem");
    host_coef = (float*) malloc (
	7 * options->image_resolution[0] * options->image_resolution[1] 
	* sizeof(float));
		
    for (a = 0; a < options->num_angles; a++)
    {

	printf(" %d\n",image_num);
	fflush(stdout);

	plm_timer_start (&timer);

#if defined (commentout)
	// Load the current image
	cbi = get_image_raw (options, image_num);
	if (cbi==NULL)
	    continue;
#endif

	// Load dynamic kernel arguments
	kargs->img_dim.x = cbi->dim[0];
	kargs->img_dim.y = cbi->dim[1];
	kargs->ic.x = cbi->pmat->ic[0];
	kargs->ic.y = cbi->pmat->ic[1];
	kargs->nrm.x = cbi->pmat->nrm[0];
	kargs->nrm.y = cbi->pmat->nrm[1];
	kargs->nrm.z = cbi->pmat->nrm[2];
	kargs->sad = cbi->pmat->sad;
	kargs->sid = cbi->pmat->sid;
	for(i=0; i<12; i++)
	    kargs->matrix[i] = (float)cbi->pmat->matrix[i];

	//Precalculate coeff

	int xy7;
	double * ic=cbi->pmat->ic;
	for (int x=0;x<cbi->dim[0];x++)
	    for (int y=0; y<cbi->dim[1];y++){
		xy7=7*(y*cbi->dim[0]+x);
		host_coef[xy7]  =((y-ic[1])*cbi->pmat->matrix[8]-cbi->pmat->matrix[4])/(cbi->pmat->matrix[5]-(y-ic[1])*cbi->pmat->matrix[9]);
		host_coef[xy7+2]=((y-ic[1])*cbi->pmat->matrix[9]-cbi->pmat->matrix[5])/(cbi->pmat->matrix[4]-(y-ic[1])*cbi->pmat->matrix[8]);
		host_coef[xy7+1]=(y-ic[1])*cbi->pmat->matrix[11]/(cbi->pmat->matrix[5]-(y-ic[1])*cbi->pmat->matrix[9]);
		host_coef[xy7+3]=(y-ic[1])*cbi->pmat->matrix[11]/(cbi->pmat->matrix[4]-(y-ic[1])*cbi->pmat->matrix[8]);
		host_coef[xy7+4]=(x-ic[0])*cbi->pmat->matrix[8]/cbi->pmat->matrix[2];
		host_coef[xy7+5]=(x-ic[0])*cbi->pmat->matrix[9]/cbi->pmat->matrix[2];
		host_coef[xy7+6]=(x-ic[0])*cbi->pmat->matrix[11]/cbi->pmat->matrix[2];
	    }

	time_io += plm_timer_report (&timer);
	plm_timer_start (&timer);

	// Copy image pixel data & projection matrix to device Global Memory
	// and then bind them to the texture hardware.
	//cudaMemcpy( dev_img, cbi->img, cbi->dim[0]*cbi->dim[1]*sizeof(float), cudaMemcpyHostToDevice );
	//cudaBindTexture( 0, tex_img, dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float) );

	//cudaMemcpy(dev_vol,  vol->img, vol->npix * vol->pix_size, cudaMemcpyHostToDevice );

	cudaMemcpy( dev_matrix, kargs->matrix, sizeof(kargs->matrix), cudaMemcpyHostToDevice );

	cudaBindTexture( 0, tex_matrix, dev_matrix, sizeof(kargs->matrix)); 

	cudaMemcpy( dev_coef, host_coef, 7*cbi->dim[0]*cbi->dim[1]*sizeof(float), cudaMemcpyHostToDevice );

	cudaBindTexture( 0, tex_coef, dev_coef,  7*cbi->dim[0]*cbi->dim[1]*sizeof(float)); 

	//cudaBindTexture( 0, tex_vol, dev_vol,  vol->npix * vol->pix_size); 

	// Free the current vol 
	//free_cb_image( cbi );

	// Thead Block Dimensions
	int tBlock_x = vol->dim[0];
	int tBlock_y = 1;
	int tBlock_z = 1;

	// Each element in the volume (each voxel) gets 1 thread
	int blocksInX = cbi->dim[0];
	int blocksInY = cbi->dim[1];
	dim3 dimGrid  = dim3(blocksInX, blocksInY);
	dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

	// Invoke ze kernel  \(^_^)/
	// Note: cbi->img AND cbi->matrix are passed via texture memory

	int smemSize = vol->dim[0]  * sizeof(float);
	//	if (abs(kargs->matrix[5])>abs(kargs->matrix[4]))

	plm_timer_start (&timer);

	//-------------------------------------
	kernel_drr_i3<<< dimGrid, dimBlock,  smemSize>>>(dev_img, 
	    kargs->img_dim,
	    kargs->ic,
	    kargs->nrm,
	    kargs->sad,
	    kargs->scale,
	    kargs->vol_offset,
	    kargs->vol_dim,
	    kargs->vol_pix_spacing);

	checkCUDAError("Kernel Panic!");

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
	cudaUnbindTexture( tex_matrix );
	cudaUnbindTexture( tex_coef);

	// Copy reconstructed volume from device to host
	//cudaMemcpy( vol->img, dev_vol, vol->npix * vol->pix_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( cbi->img, dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float), cudaMemcpyDeviceToHost );
	checkCUDAError("Error: Unable to retrieve data volume.");
		
	char img_file[1024];
	
	size_t rc;
	FILE* fp;
	//sprintf (fmt, "%s\\%s\\%s", options->input_dir,options->sub_dir,img_file_pat);
	//sprintf (fmt, "%s\\%s", options->input_dir,img_file_pat);
	//   sprintf (img_file, fmt, image_num);
	//   sprintf (fmt, "%s\\%s", options->input_dir, mat_file_pat);
	//   sprintf (mat_file, fmt, image_num);
	//   return load_and_filter_cb_image (options,img_file, mat_file);
	

	//sprintf (img_file, "%s/DRR/Proj_%03d.raw", options->input_dir,image_num);

	printf ("GCS Warning: output filename may not be flexible enough\n");
	sprintf (img_file, "%s%04d.txt", options->output_prefix, a);

	//   sprintf (img_file, fmt, image_num);
	//   sprintf (fmt, "%s\\%s", options->input_dir, mat_file_pat);

	fp = fopen (img_file,"wb");
	if (!fp) {
	    fprintf (stderr, "Can't open file %s for write\n. Skipped", img_file);
	    return(1);
	}
	float writeimg[512*384];
	for (int i=0; i<512*384; i++)
	    writeimg[i]=65535*exp(-cbi->img[i]/30000);
	//		writeimg[i]=cbi->img[i];

	/* write pixels */
	rc = fwrite (writeimg , sizeof(float),  512* 384, fp); 
	if (rc != 512 * 384) {
	    fprintf (stderr, "Couldn't write raster data for %s\n",
		img_file);
	    return(1);
	}
	printf("Writing OK\n");
			
	fclose(fp);
	proj_image_free ( cbi );

    }

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif

    // Report Timing Data
    time_total = plm_timer_report (&total_timer);
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", time_total);
#if defined (TIME_KERNEL)
    printf ("\tTotal Kernel  Time: %.9fs\n", time_kernel);
    printf ("\tTotal File IO Time: %.9fs\n\n", time_io);
#endif

    printf ("[Average Projection Time: %.9fs ]\n", 
	time_total / options->num_angles);
#if defined (TIME_KERNEL)
    printf ("\tAverage Kernel  Time: %.9fs\n", 
	time_kernel / options->num_angles);
    printf ("\tAverage File IO Time: %.9fs\n\n", 
	time_io / options->num_angles);
#endif
    printf("========================================\n");

    // Cleanup
    cudaFree( dev_img );
    cudaFree( dev_kargs );
    cudaFree( dev_matrix );
    //cudaFree( dev_vol );
    cudaFree( dev_coef);
    free(host_coef);

    return 0;
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: checkCUDAError() /////////////////////////////////////////////
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err) 
    {
        fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// Vim Editor Settings ////////////////////////////////////////////////////
// vim:ts=8:sw=8:cindent:nowrap
///////////////////////////////////////////////////////////////////////////
