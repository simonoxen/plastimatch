/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#define _CRT_SECURE_NO_DEPRECATE
#define READ_PFM
#define WRITE_BLOCK (1024*1024)

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

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>

/*****************
* FDK  #includes *
*****************/
#include "fdk_cuda_p.h"
#include "fdk_opts.h"
#include "fdk_utils.h"
#include "math_util.h"
#include "proj_image.h"
#include "readmha.h"
#include "volume.h"

/*********************
* High Res Win Timer *
*********************/
#include "timer.h"
#if defined (_WIN32)
#include <windows.h>
#include <winbase.h>
#endif


// P R O T O T Y P E S ////////////////////////////////////////////////////
void checkCUDAError(const char *msg);

__global__ void kernel_fdk (float *dev_vol, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing, unsigned int Blocks_Y, float invBlocks_Y);

__global__ void kernel_fdk2 (float *dev_vol, int img_num, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing, unsigned int Blocks_Y, float invBlocks_Y);

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
void kernel_fdk (float *dev_vol, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing, unsigned int Blocks_Y, float invBlocks_Y)
{
	// CUDA 2.0 does not allow for a 3D grid, which severely
	// limits the manipulation of large 3D arrays of data.  The
	// following code is a hack to bypass this implementation
	// limitation.
	unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
	unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
	unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
	unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

	if( i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z )
		return; 
/*
	// EXPERIMENTAL //////////////////////////////////////////////
	long int one_block = blockDim.x * blockDim.y * blockDim.z;
	long int offset_x  = blockIdx.x * (one_block);
	long long int offset_y  = blockIdx.y * (one_block * gridDim.x);
	long int block_index = threadIdx.x + (blockDim.x * threadIdx.y) + (blockDim.x * blockDim.y * threadIdx.z);
	long int vol_idx = offset_x + offset_y + block_index;
	//////////////////////////////////////////////////////////////
*/
	// Index row major into the volume
	long int vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );

	float3 vp;
	float3 ip;
	float  s;
	float voxel_data;


	// offset volume coords
	vp.x = vol_offset.x + i * vol_pix_spacing.x;	// Compiler should combine into 1 FMAD.
	vp.y = vol_offset.y + j * vol_pix_spacing.y;	// Compiler should combine into 1 FMAD.
	vp.z = vol_offset.z + k * vol_pix_spacing.z;	// Compiler should combine into 1 FMAD.

	// matrix multiply
	ip.x = tex1Dfetch(tex_matrix, 0)*vp.x + tex1Dfetch(tex_matrix, 1)*vp.y + tex1Dfetch(tex_matrix, 2)*vp.z + tex1Dfetch(tex_matrix, 3);
	ip.y = tex1Dfetch(tex_matrix, 4)*vp.x + tex1Dfetch(tex_matrix, 5)*vp.y + tex1Dfetch(tex_matrix, 6)*vp.z + tex1Dfetch(tex_matrix, 7);
	ip.z = tex1Dfetch(tex_matrix, 8)*vp.x + tex1Dfetch(tex_matrix, 9)*vp.y + tex1Dfetch(tex_matrix, 10)*vp.z + tex1Dfetch(tex_matrix, 11);

	// Change coordinate systems
	ip.x = ic.x + ip.x / ip.z;
	ip.y = ic.y + ip.y / ip.z;

	// Get pixel from 2D image
	ip.x = __float2int_rd(ip.x);
	ip.y = __float2int_rd(ip.y);
	voxel_data = tex1Dfetch(tex_img, ip.x*img_dim.y + ip.y);

	// Dot product
	s = nrm.x*vp.x + nrm.y*vp.y + nrm.z*vp.z;

	// Conebeam weighting factor
	s = sad - s;
	s = (sad * sad) / (s * s);

	// Place it into the volume
	dev_vol[vol_idx] += scale * s * voxel_data;
}
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_


//The alternative method of CUDA_reconstruct_conebeam 
//load all projection image into texture memory
//use pixel-driven method to backproject projection data to the volume pixel.
//Use sum reduction to add all pixel togethers.
//Although this method achieves similar projection/second speed as the origional method
//The memory liite of CUDA applicaiton seems to limit the maximum number of projecitons 
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
__global__
void kernel_fdk2 (float *dev_vol, int img_num, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing, unsigned int Blocks_Y, float invBlocks_Y)
{
	// CUDA 2.0 does not allow for a 3D grid, which severely
	// limits the manipulation of large 3D arrays of data.  The
	// following code is a hack to bypass this implementation
	// limitation.
	extern __shared__ float sdata[];
	unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
	unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
	unsigned int i = blockIdx.x;
	unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
	unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

	unsigned int tid=threadIdx.x;

	if( i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z )
		return; 
/*
	// EXPERIMENTAL //////////////////////////////////////////////
	long int one_block = blockDim.x * blockDim.y * blockDim.z;
	long int offset_x  = blockIdx.x * (one_block);
	long long int offset_y  = blockIdx.y * (one_block * gridDim.x);
	long int block_index = threadIdx.x + (blockDim.x * threadIdx.y) + (blockDim.x * blockDim.y * threadIdx.z);
	long int vol_idx = offset_x + offset_y + block_index;
	//////////////////////////////////////////////////////////////
*/
	// Index row major into the volume
	long int vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );

	float3 vp;
	float3 ip;
	float  s;
	float voxel_data;


	// offset volume coords
	vp.x = vol_offset.x + i * vol_pix_spacing.x;	// Compiler should combine into 1 FMAD.
	vp.y = vol_offset.y + j * vol_pix_spacing.y;	// Compiler should combine into 1 FMAD.
	vp.z = vol_offset.z + k * vol_pix_spacing.z;	// Compiler should combine into 1 FMAD.
	
	unsigned int Proj12=tid*12;
	// matrix multiply
	ip.x = tex1Dfetch(tex_matrix, 0+Proj12)*vp.x + tex1Dfetch(tex_matrix, 1+Proj12)*vp.y + tex1Dfetch(tex_matrix, 2+Proj12)*vp.z + tex1Dfetch(tex_matrix, 3+Proj12);
	ip.y = tex1Dfetch(tex_matrix, 4+Proj12)*vp.x + tex1Dfetch(tex_matrix, 5+Proj12)*vp.y + tex1Dfetch(tex_matrix, 6+Proj12)*vp.z + tex1Dfetch(tex_matrix, 7+Proj12);
	ip.z = tex1Dfetch(tex_matrix, 8+Proj12)*vp.x + tex1Dfetch(tex_matrix, 9+Proj12)*vp.y + tex1Dfetch(tex_matrix, 10+Proj12)*vp.z + tex1Dfetch(tex_matrix, 11+Proj12);

	// Change coordinate systems
	ip.x = ic.x + ip.x / ip.z;
	ip.y = ic.y + ip.y / ip.z;

	// Get pixel from 2D image
	ip.x = __float2int_rd(ip.x);
	ip.y = __float2int_rd(ip.y);
	voxel_data = tex1Dfetch(tex_img, tid*img_dim.x*img_dim.y+ip.x*img_dim.y + ip.y);
	//voxel_data = tex1Dfetch(tex_img, tid+(ip.x*img_dim.y + ip.y)*img_num);

	// Dot product
	s = nrm.x*vp.x + nrm.y*vp.y + nrm.z*vp.z;

	// Conebeam weighting factor
	s = sad - s;
	s = (sad * sad) / (s * s);

	// Place it into the volume
	sdata[tid]= scale * s * voxel_data;


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
		dev_vol[vol_idx]= sdata[0];
}
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

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
	unsigned int ui;
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

#if 1
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
#endif


//DRR3  uses 3D textures and pre-calculated coefs to accelerate DRR generation.

void kernel_drr_i3 (float * dev_img, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing)
{
	// CUDA 2.0 does not allow for a 3D grid, which severely
	// limits the manipulation of large 3D arrays of data.  The
	// following code is a hack to bypass this implementation
	// limitation.
	extern __shared__ float sdata[];
	float3 ip;
	float3 vp;
	int i,j,k;
	int x,y,xy7;
	unsigned int ui;
	long int vol_idx;
	float vol;

	unsigned int tid = threadIdx.x;

	x = blockIdx.x;
	y = blockIdx.y;
	xy7=7*(y*img_dim.x+x);
	
	if (abs(tex1Dfetch(tex_matrix, 5))>abs(tex1Dfetch(tex_matrix, 4))){

		vp.x=vol_offset.x+threadIdx.x*vol_pix_spacing.x;

		vp.y=tex1Dfetch(tex_coef, xy7)*vp.x+tex1Dfetch(tex_coef, xy7+1);

		vp.z=tex1Dfetch(tex_coef, xy7+4)*vp.x+tex1Dfetch(tex_coef, xy7+5)*vp.y+tex1Dfetch(tex_coef, xy7+6);

		i=  threadIdx.x;

		j=  __float2int_rd((vp.y-vol_offset.y)/vol_pix_spacing.y);

		k=  __float2int_rd((vp.z-vol_offset.z)/vol_pix_spacing.z);

		//if (j<0||j>=vol_dim.y||k<0||k>=vol_dim.z)
		if ((i-vol_dim.x/2)*(i-vol_dim.x/2)+(j-vol_dim.y/2)*(j-vol_dim.y/2)>vol_dim.y*vol_dim.y/4||k<0||k>=vol_dim.z)
			sdata[tid]=0.0f;
		else{
			vol=tex3D(tex_3Dvol,i,j,k);
	
			sdata[tid]=(vol+1000.0f);

		}
	}
	else{
	
		vp.y=vol_offset.y+threadIdx.x*vol_pix_spacing.y;

		vp.x=tex1Dfetch(tex_coef, xy7+2)*vp.y+tex1Dfetch(tex_coef, xy7+3);

		vp.z=tex1Dfetch(tex_coef, xy7+4)*vp.x+tex1Dfetch(tex_coef, xy7+5)*vp.y+tex1Dfetch(tex_coef, xy7+6);

		j=  threadIdx.x;

		i=  __float2int_rd((vp.x-vol_offset.x)/vol_pix_spacing.x);

		k=  __float2int_rd((vp.z-vol_offset.z)/vol_pix_spacing.z);

		if ((i-vol_dim.x/2)*(i-vol_dim.x/2)+(j-vol_dim.y/2)*(j-vol_dim.y/2)>vol_dim.y*vol_dim.y/4||k<0||k>=vol_dim.z)
			sdata[tid]=0.0f;
		else{
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

    // do reduction in shared mem





    // write result for this block to global mem
    if (tid == 0) 
		dev_img[blockIdx.x*img_dim.y + blockIdx.y] = sdata[0];


}



//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_



///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam() //////////////////////////////////
extern "C"
int CUDA_reconstruct_conebeam_ext (Volume *vol, Fdk_options *options)
{
    // Thead Block Dimensions
    int tBlock_x = 16;
    int tBlock_y = 4;
    int tBlock_z = 4;

    // Each element in the volume (each voxel) gets 1 thread
    int blocksInX = (vol->dim[0]+tBlock_x-1)/tBlock_x;
    int blocksInY = (vol->dim[1]+tBlock_y-1)/tBlock_y;
    int blocksInZ = (vol->dim[2]+tBlock_z-1)/tBlock_z;
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Size of volume Malloc
    int vol_size_malloc = (vol->dim[0]*vol->dim[1]*vol->dim[2])*sizeof(float);

    printf ("Hello world from CUDA_reconstruct_conebeam_ext\n");

    // Structure for passing arugments to kernel: (See fdk_cuda.h)
    kernel_args_fdk *kargs;
    kargs = (kernel_args_fdk *) malloc(sizeof(kernel_args_fdk));

    Proj_image* cbi;
    int image_num;
    int i;

    // CUDA device pointers
    float *dev_vol;	            // Holds voxels on device
    float *dev_img;	            // Holds image pixels on device
    float *dev_matrix;
    kernel_args_fdk *dev_kargs; // Holds kernel parameters
    cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
    checkCUDAError ("cudaMalloc");
    cudaMalloc( (void**)&dev_kargs, sizeof(kernel_args_fdk) );
    checkCUDAError ("cudaMalloc");

    printf ("Calculating scale: %d\n", options->skip_img);
    // Calculate the scale
    image_num = 1 + (options->last_img - options->first_img) / options->skip_img;
    float scale = (float) (sqrt(3.0) / (double) image_num);
    scale = scale * options->scale;

    printf ("loading static kernel arguments\n");

    // Load static kernel arguments
    kargs->scale = scale;
    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];


    ////// TIMING CODE //////////////////////
    // Initialize Windows HighRes Timer
    int count,a;
    float kernel_total, io_total;
#if defined (_WIN32)
    LARGE_INTEGER ticksPerSecond;
    LARGE_INTEGER tick;   // A point in time
    LARGE_INTEGER start_ticks_kernel, end_ticks_kernel, cputime;   
    LARGE_INTEGER start_ticks_io, end_ticks_io;   
    LARGE_INTEGER start_ticks_total, end_ticks_total;   
#endif

#if defined (VERBOSE)
    printf("\n\nInitializing High Resolution Timers\n");
#endif
    io_total=0;
    kernel_total=0;

    // get the high resolution counter's accuracy
#if defined (_WIN32)
    if (!QueryPerformanceFrequency(&ticksPerSecond))
	printf("QueryPerformance not present!");
#endif

    // Test: Get current time.
#if defined (_WIN32)
    if (!QueryPerformanceCounter(&tick))
	printf("no go counter not installed");  
#endif

    /////////////////////////////////////////
	

#if defined (VERBOSE)
    // First, we need to allocate memory on the host device
    // for the 3D volume of voxels that will hold our reconstruction.
    printf("========================================\n");
    printf("Allocating %dMB of video memory...", vol_size_malloc/1000000);
#endif

    cudaMalloc( (void**)&dev_vol, vol_size_malloc);
    cudaMemset( (void *) dev_vol, 0, vol_size_malloc);	
    checkCUDAError("Unable to allocate data volume");

#if defined (VERBOSE)
    printf(" done.\n\n");

    // State the kernel execution parameters
    printf("kernel parameters:\n dimGrid: %u, %u (Logical: %u, %u, %u)\n dimBlock: %u, %u, %u\n", dimGrid.x, dimGrid.y, dimGrid.x, blocksInY, blocksInZ, dimBlock.x, dimBlock.y, dimBlock.z);
    printf("%u voxels in volume\n", vol->npix);
    printf("%u projections to process\n", 1+(options->last_img - options->first_img) / options->skip_img);
    printf("%u Total Operations\n", vol->npix * (1+(options->last_img - options->first_img) / options->skip_img));
    printf("========================================\n\n");

    // Start working
    printf("Processing...");
#endif

	
	
    // This is just to retrieve the 2D image dimensions
    int fimg=options->first_img;
    do {
	printf ("Calling get_image\n");
	cbi = get_image_raw (options, fimg);
	fimg++;
    }
    while(cbi==NULL);
		
    cudaMalloc( (void**)&dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float)); 
    proj_image_free( cbi );

    ////// TIMING CODE //////////////////////
#if defined (_WIN32)
    QueryPerformanceCounter(&start_ticks_total);
#endif
    /////////////////////////////////////////

    printf ("Projecting Image:");
    // Project each image into the volume one at a time
    for(image_num = options->first_img;  image_num <= options->last_img;  image_num += options->skip_img)
    {
	printf (" %d\n", image_num);
	fflush(stdout);
	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&start_ticks_io);
#endif
#endif
	/////////////////////////////////////////

	// Load the current image
	cbi = get_image_raw (options, image_num);
	if (cbi==NULL)
	    continue;

	// Load dynamic kernel arguments
	kargs->img_dim.x = cbi->dim[0];
	kargs->img_dim.y = cbi->dim[1];
	kargs->ic.x = cbi->ic[0];
	kargs->ic.y = cbi->ic[1];
	kargs->nrm.x = cbi->nrm[0];
	kargs->nrm.y = cbi->nrm[1];
	kargs->nrm.z = cbi->nrm[2];
	kargs->sad = cbi->sad;
	kargs->sid = cbi->sid;
	for(i=0; i<12; i++)
	    kargs->matrix[i] = (float)cbi->matrix[i];

	// Copy image pixel data & projection matrix to device Global Memory
	// and then bind them to the texture hardware.
	cudaMemcpy( dev_img, cbi->img, cbi->dim[0]*cbi->dim[1]*sizeof(float), cudaMemcpyHostToDevice );
	cudaBindTexture( 0, tex_img, dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float) );
	cudaMemcpy( dev_matrix, kargs->matrix, sizeof(kargs->matrix), cudaMemcpyHostToDevice );
	cudaBindTexture( 0, tex_matrix, dev_matrix, sizeof(kargs->matrix));

	// Free the current image 
	proj_image_free( cbi );

	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&end_ticks_io);
	cputime.QuadPart = end_ticks_io.QuadPart- start_ticks_io.QuadPart;
	io_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
	QueryPerformanceCounter(&start_ticks_kernel);
#endif
#endif
	/////////////////////////////////////////

	// Invoke ze kernel  \(^_^)/
	// Note: cbi->img AND cbi->matrix are passed via texture memory
	//-------------------------------------
	kernel_fdk<<< dimGrid, dimBlock >>>(dev_vol,
					    kargs->img_dim,
					    kargs->ic,
					    kargs->nrm,
					    kargs->sad,
					    kargs->scale,
					    kargs->vol_offset,
					    kargs->vol_dim,
					    kargs->vol_pix_spacing,
					    blocksInY,
					    1.0f/(float)blocksInY);
	checkCUDAError("Kernel Panic!");

#if defined (TIME_KERNEL)
	// CUDA kernel calls are asynchronous...
	// In order to accurately time the kernel
	// execution time we need to set a thread
	// barrier here after its execution.
	cudaThreadSynchronize();
#endif

	// Unbind the image and projection matrix textures
	cudaUnbindTexture( tex_img );
	cudaUnbindTexture( tex_matrix );

	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&end_ticks_kernel);
	cputime.QuadPart = end_ticks_kernel.QuadPart- start_ticks_kernel.QuadPart;
	kernel_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
#endif
#endif
	/////////////////////////////////////////
    }

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif
	
    // Copy reconstructed volume from device to host
    cudaMemcpy( vol->img, dev_vol, vol->npix * vol->pix_size, cudaMemcpyDeviceToHost );
    checkCUDAError("Error: Unable to retrieve data volume.");

	
    ////// TIMING CODE //////////////////////
    // Report Timing Data
#if defined (_WIN32)
#if defined (TIME_KERNEL)
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf ("\tTotal Kernel  Time: %.9fs\n", kernel_total);
    printf ("\tTotal File IO Time: %.9fs\n\n", io_total);

    printf ("[Average Projection Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart)/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage Kernel  Time: %.9fs\n", kernel_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage File IO Time: %.9fs\n\n", io_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf("========================================\n");
#else
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf("========================================\n");
#endif
#endif
    /////////////////////////////////////////


    // Cleanup
    cudaFree( dev_img );
    cudaFree( dev_kargs );
    cudaFree( dev_matrix );
    cudaFree( dev_vol );	

    return 0;
}
//}
///////////////////////////////////////////////////////////////////////////


//The alternative method of CUDA_reconstruct_conebeam 
//load all projection image into texture memory
//use pixel-driven method to backproject projection data to the volume pixel.
//Use sum reduction to add all pixel togethers.
//Although this method achieves similar projection/second speed as the origional method
//The memory liite of CUDA applicaiton seems to limit the maximum number of projecitons 

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam() //////////////////////////////////
extern "C"
int CUDA_reconstruct_conebeam2 (Volume *vol, Fdk_options *options)
{



    // Size of volume Malloc
    int vol_size_malloc = (vol->dim[0]*vol->dim[1]*vol->dim[2])*sizeof(float);

    // Structure for passing arugments to kernel: (See fdk_cuda.h)
    kernel_args_fdk *kargs;
    kargs = (kernel_args_fdk *) malloc(sizeof(kernel_args_fdk));

    Proj_image* cbi;
    int image_num, image_idx, image_n;
    int i;

    // CUDA device pointers
    float *dev_vol;	            // Holds voxels on device
    float *host_img;
    float *dev_img;	            // Holds image pixels on device
    float *host_matrix;
    float *dev_matrix;
    kernel_args_fdk *dev_kargs; // Holds kernel parameters


    // Calculate the scale
    image_n = 1 + (options->last_img - options->first_img) / options->skip_img;
    float scale = (float) (sqrt(3.0) / (double) image_n);
    scale = scale * options->scale;





    // Load static kernel arguments
    kargs->scale = scale;
    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];


    ////// TIMING CODE //////////////////////
    // Initialize Windows HighRes Timer
    int count,a;
    float kernel_total, io_total;
#if defined (_WIN32)
    LARGE_INTEGER ticksPerSecond;
    LARGE_INTEGER tick;   // A point in time
    LARGE_INTEGER start_ticks_kernel, end_ticks_kernel, cputime;   
    LARGE_INTEGER start_ticks_io, end_ticks_io;   
    LARGE_INTEGER start_ticks_total, end_ticks_total;   
#endif

#if defined (VERBOSE)
    printf("\n\nInitializing High Resolution Timers\n");
#endif
    io_total=0;
    kernel_total=0;

    // get the high resolution counter's accuracy
#if defined (_WIN32)
    if (!QueryPerformanceFrequency(&ticksPerSecond))
	printf("QueryPerformance not present!");
#endif

    // Test: Get current time.
#if defined (_WIN32)
    if (!QueryPerformanceCounter(&tick))
	printf("no go counter not installed");  
#endif


#if defined (VERBOSE)
    // First, we need to allocate memory on the host device
    // for the 3D volume of voxels that will hold our reconstruction.
    printf("========================================\n");
    printf("Allocating %dMB of video memory...", vol_size_malloc/1000000);
#endif


#if defined (VERBOSE)
    printf(" done.\n\n");

    // Start working
    printf("Processing...");
#endif

	
	
    // This is just to retrieve the 2D image dimensions
    int fimg=options->first_img;
    do{
	cbi = get_image_raw (options, fimg);
	fimg++;
    }
    while(cbi==NULL);


    cudaMalloc( (void**)&dev_matrix, 12*sizeof(float)*image_n );
    checkCUDAError("Unable to allocate dev_matrix");
    
    cudaMalloc( (void**)&dev_kargs, sizeof(kernel_args_fdk) );
    checkCUDAError("Unable to allocate dev_kargs");

    cudaMalloc( (void**)&dev_vol, vol_size_malloc);
    cudaMemset( (void *) dev_vol, 0, vol_size_malloc);	
    checkCUDAError("Unable to allocate dev_vol");

		
    cudaMalloc( (void**)&dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float)*image_n); 
    checkCUDAError("Unable to allocate dev_img");

    host_matrix=(float *)malloc(12*sizeof(float)*image_n);
    if (host_matrix==NULL){
	printf("Can no allocate host memory for host_matrix");
	exit(1);
    }
    host_img=(float *)malloc(cbi->dim[0]*cbi->dim[1]*sizeof(float)*image_n); 
    if (host_img==NULL){
	printf("Can no allocate host memory for host_img");
	exit(1);
    }
    proj_image_free( cbi );

    ////// TIMING CODE //////////////////////
#if defined (_WIN32)
    QueryPerformanceCounter(&start_ticks_total);
#endif
    /////////////////////////////////////////

    printf ("Projecting Image:");
    image_idx=0;
    // Project each image into the volume one at a time
    for(image_num = options->first_img;  image_num <= options->last_img;  image_num += options->skip_img)
    {
	printf (" %d\n", image_num);
	fflush(stdout);
	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&start_ticks_io);
#endif
#endif
	/////////////////////////////////////////

	// Load the current image
	cbi = get_image_raw (options, image_num);
	if (cbi==NULL)
	    continue;


	// Load dynamic kernel arguments
	kargs->img_dim.x = cbi->dim[0];
	kargs->img_dim.y = cbi->dim[1];
	kargs->ic.x = cbi->ic[0];
	kargs->ic.y = cbi->ic[1];
	kargs->nrm.x = cbi->nrm[0];
	kargs->nrm.y = cbi->nrm[1];
	kargs->nrm.z = cbi->nrm[2];
	kargs->sad = cbi->sad;
	kargs->sid = cbi->sid;
	for(i=0; i<12; i++)
	    kargs->matrix[i] = (float)cbi->matrix[i];


	memcpy(host_img+image_idx*cbi->dim[0]*cbi->dim[1],cbi->img, cbi->dim[0]*cbi->dim[1]*sizeof(float));
	float *cbi_img=cbi->img;
	//for (i=0; i<cbi->dim[0]*cbi->dim[1]; i++)
	//	host_img[image_idx+i*image_n]=cbi_img[i];

	memcpy(host_matrix+image_idx*12,kargs->matrix,12*sizeof(float));


	// Free the current image 
	proj_image_free( cbi );
	image_idx++;

    }

	
    // Copy image pixel data & projection matrix to device Global Memory
    // and then bind them to the texture hardware.
    cudaMemcpy( dev_img, host_img, kargs->img_dim.x*kargs->img_dim.y*sizeof(float)*image_n, cudaMemcpyHostToDevice );
    printf("Copy imgs from host to device");
    cudaBindTexture( 0, tex_img, dev_img, kargs->img_dim.x*kargs->img_dim.y*sizeof(float)*image_n);
    cudaMemcpy( dev_matrix, host_matrix, sizeof(kargs->matrix)*image_n, cudaMemcpyHostToDevice);
    cudaBindTexture( 0, tex_matrix, dev_matrix, sizeof(kargs->matrix)*image_n);

    free(host_img);
    free(host_matrix);

    ////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
    QueryPerformanceCounter(&end_ticks_io);
    cputime.QuadPart = end_ticks_io.QuadPart- start_ticks_io.QuadPart;
    io_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
    QueryPerformanceCounter(&start_ticks_kernel);
#endif
#endif
    /////////////////////////////////////////

    int smemSize = image_n  * sizeof(float);

    // Thead Block Dimensions
    int tBlock_x = image_n;
    int tBlock_y = 1;
    int tBlock_z = 1;

    // Each element in the volume (each voxel) gets 1 thread
    int blocksInX = vol->dim[0];
    int blocksInY = vol->dim[1];
    int blocksInZ = vol->dim[2];
    dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);


    // Invoke ze kernel  \(^_^)/
    // Note: cbi->img AND cbi->matrix are passed via texture memory
    //-------------------------------------

    kernel_fdk2<<< dimGrid, dimBlock, smemSize >>>(dev_vol, 
	image_n,
	kargs->img_dim,
	kargs->ic,
	kargs->nrm,
	kargs->sad,
	kargs->scale,
	kargs->vol_offset,
	kargs->vol_dim,
	kargs->vol_pix_spacing,
	blocksInY,
	1.0f/(float)blocksInY);
    checkCUDAError("Kernel Panic!");

#if defined (TIME_KERNEL)
    // CUDA kernel calls are asynchronous...
    // In order to accurately time the kernel
    // execution time we need to set a thread
    // barrier here after its execution.
    cudaThreadSynchronize();
#endif

    // Unbind the image and projection matrix textures
    cudaUnbindTexture( tex_img );
    cudaUnbindTexture( tex_matrix );

    ////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
    QueryPerformanceCounter(&end_ticks_kernel);
    cputime.QuadPart = end_ticks_kernel.QuadPart- start_ticks_kernel.QuadPart;
    kernel_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
#endif
#endif
    /////////////////////////////////////////
 
#if defined (VERBOSE)
    printf(" done.\n\n");
#endif
	
    // Copy reconstructed volume from device to host
    cudaMemcpy( vol->img, dev_vol, vol->npix * vol->pix_size, cudaMemcpyDeviceToHost );
    checkCUDAError("Error: Unable to retrieve data volume.");

	
    ////// TIMING CODE //////////////////////
    // Report Timing Data
#if defined (_WIN32)
#if defined (TIME_KERNEL)
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf ("\tTotal Kernel  Time: %.9fs\n", kernel_total);
    printf ("\tTotal File IO Time: %.9fs\n\n", io_total);

    printf ("[Average Projection Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart)/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage Kernel  Time: %.9fs\n", kernel_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage File IO Time: %.9fs\n\n", io_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf("========================================\n");
#else
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf("========================================\n");
#endif
#endif
    /////////////////////////////////////////


    // Cleanup
    cudaFree( dev_img );
    cudaFree( dev_kargs );
    cudaFree( dev_matrix );
    cudaFree( dev_vol );	

    return 0;
}
//}
///////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_DRR() //////////////////////////////////
extern "C"
int CUDA_DRR (Volume *vol, Fdk_options *options)
{
    //// Thead Block Dimensions
    //int tBlock_x = 16;
    //int tBlock_y = 4;
    //int tBlock_z = 4;

    //// Each element in the volume (each voxel) gets 1 thread
    //int blocksInX = (vol->dim[0]+tBlock_x-1)/tBlock_x;
    //int blocksInY = (vol->dim[1]+tBlock_y-1)/tBlock_y;
    //int blocksInZ = (vol->dim[2]+tBlock_z-1)/tBlock_z;
    //dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    //dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Size of volume Malloc
    //int vol_size_malloc = (vol->dim[0]*vol->dim[1]*vol->dim[2])*sizeof(float);

    // Structure for passing arugments to kernel: (See fdk_cuda.h)
    kernel_args_fdk *kargs;
    kargs = (kernel_args_fdk *) malloc(sizeof(kernel_args_fdk));

    Proj_image* cbi;
    int image_num;
    int i;

    // CUDA device pointers
    float *dev_vol;	            // Holds voxels on device
    float *dev_img;	            // Holds image pixels on device
    float *dev_matrix;
    kernel_args_fdk *dev_kargs; // Holds kernel parameters
    cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
    cudaMalloc( (void**)&dev_kargs, sizeof(kernel_args_fdk) );

    // Calculate the scale
    image_num = 1 + (options->last_img - options->first_img) / options->skip_img;
    float scale = (float) (sqrt(3.0) / (double) image_num);
    scale = scale * options->scale;

    // Load static kernel arguments
    kargs->scale = scale;
    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];


    ////// TIMING CODE //////////////////////
    // Initialize Windows HighRes Timer
    int count,a;
    float kernel_total, io_total;
#if defined (_WIN32)
    LARGE_INTEGER ticksPerSecond;
    LARGE_INTEGER tick;   // A point in time
    LARGE_INTEGER start_ticks_kernel, end_ticks_kernel, cputime;   
    LARGE_INTEGER start_ticks_io, end_ticks_io;   
    LARGE_INTEGER start_ticks_total, end_ticks_total;   
#endif

#if defined (VERBOSE)
    printf("\n\nInitializing High Resolution Timers\n");
#endif
    io_total=0;
    kernel_total=0;

    // get the high resolution counter's accuracy
#if defined (_WIN32)
    if (!QueryPerformanceFrequency(&ticksPerSecond))
	printf("QueryPerformance not present!");
#endif

    // Test: Get current time.
#if defined (_WIN32)
    if (!QueryPerformanceCounter(&tick))
	printf("no go counter not installed");  
#endif

    /////////////////////////////////////////
	

    cudaMalloc( (void**)&dev_vol, vol->npix*sizeof(float));
    //cudaMemset( (void *) dev_vol, 0, vol_size_malloc);	
    checkCUDAError("Unable to allocate data volume");

	
    // This is just to retrieve the 2D image dimensions
    int fimg=options->first_img;
    do{
	cbi = get_image_raw (options, fimg);
	fimg++;
    }
    while(cbi==NULL);
		
    cudaMalloc( (void**)&dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float)); 
    proj_image_free( cbi );

    ////// TIMING CODE //////////////////////
#if defined (_WIN32)
    QueryPerformanceCounter(&start_ticks_total);
#endif
    /////////////////////////////////////////

    printf ("Projecting Image:");
    // Project each image into the volume one at a time
    for(image_num = options->first_img;  image_num <= options->last_img;  image_num += options->skip_img)
	{

	    fflush(stdout);
	    ////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	    QueryPerformanceCounter(&start_ticks_io);
#endif
#endif
	    /////////////////////////////////////////

	    // Load the current image
	    cbi = get_image_raw (options, image_num);
	    if (cbi==NULL)
		continue;

	    // Load dynamic kernel arguments
	    kargs->img_dim.x = cbi->dim[0];
	    kargs->img_dim.y = cbi->dim[1];
	    kargs->ic.x = cbi->ic[0];
	    kargs->ic.y = cbi->ic[1];
	    kargs->nrm.x = cbi->nrm[0];
	    kargs->nrm.y = cbi->nrm[1];
	    kargs->nrm.z = cbi->nrm[2];
	    kargs->sad = cbi->sad;
	    kargs->sid = cbi->sid;
	    for(i=0; i<12; i++)
		kargs->matrix[i] = (float)cbi->matrix[i];

	    // Copy image pixel data & projection matrix to device Global Memory
	    // and then bind them to the texture hardware.
	    //cudaMemcpy( dev_img, cbi->img, cbi->dim[0]*cbi->dim[1]*sizeof(float), cudaMemcpyHostToDevice );
	    //cudaBindTexture( 0, tex_img, dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float) );

	    cudaMemcpy(dev_vol,  vol->img, vol->npix * vol->pix_size, cudaMemcpyHostToDevice );

	    cudaMemcpy( dev_matrix, kargs->matrix, sizeof(kargs->matrix), cudaMemcpyHostToDevice );
	    cudaBindTexture( 0, tex_matrix, dev_matrix, sizeof(kargs->matrix)); 

	    // Free the current vol 
	    //free_cb_image( cbi );

	    ////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	    QueryPerformanceCounter(&end_ticks_io);
	    cputime.QuadPart = end_ticks_io.QuadPart- start_ticks_io.QuadPart;
	    io_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
	    QueryPerformanceCounter(&start_ticks_kernel);
#endif
#endif
	    /////////////////////////////////////////


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



	    ////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	    QueryPerformanceCounter(&end_ticks_kernel);
	    cputime.QuadPart = end_ticks_kernel.QuadPart- start_ticks_kernel.QuadPart;
	    kernel_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
#endif
#endif
	    /////////////////////////////////////////
	}

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif
	

	

	
    ////// TIMING CODE //////////////////////
    // Report Timing Data
#if defined (_WIN32)
#if defined (TIME_KERNEL)
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf ("\tTotal Kernel  Time: %.9fs\n", kernel_total);
    printf ("\tTotal File IO Time: %.9fs\n\n", io_total);

    printf ("[Average Projection Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart)/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage Kernel  Time: %.9fs\n", kernel_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage File IO Time: %.9fs\n\n", io_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf("========================================\n");
#else
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf("========================================\n");
#endif
#endif
    /////////////////////////////////////////


    // Cleanup
    cudaFree( dev_img );
    cudaFree( dev_kargs );
    cudaFree( dev_matrix );
    cudaFree( dev_vol );	

    return 0;
}

//DRR3 uses 3D textures and pre-calculated coefs to accelerate DRR generation.

//////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_DRR() //////////////////////////////////
extern "C"
int CUDA_DRR3 (Volume *vol, Fdk_options *options)
{
    //// Thead Block Dimensions
    //int tBlock_x = 16;
    //int tBlock_y = 4;
    //int tBlock_z = 4;

    //// Each element in the volume (each voxel) gets 1 thread
    //int blocksInX = (vol->dim[0]+tBlock_x-1)/tBlock_x;
    //int blocksInY = (vol->dim[1]+tBlock_y-1)/tBlock_y;
    //int blocksInZ = (vol->dim[2]+tBlock_z-1)/tBlock_z;
    //dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    //dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    // Size of volume Malloc
    //int vol_size_malloc = (vol->dim[0]*vol->dim[1]*vol->dim[2])*sizeof(float);

    // Structure for passing arugments to kernel: (See fdk_cuda.h)
    kernel_args_fdk *kargs;
    kargs = (kernel_args_fdk *) malloc(sizeof(kernel_args_fdk));

    Proj_image* cbi;
    int image_num;
    int i,j,k;


    // CUDA device pointers
    float *dev_vol;	            // Holds voxels on device
    float *dev_img;	            // Holds image pixels on device
    float *dev_matrix;
    float *dev_coef;
    float *host_coef;
    kernel_args_fdk *dev_kargs; // Holds kernel parameters
    cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
    cudaMalloc( (void**)&dev_kargs, sizeof(kernel_args_fdk) );

	


    // Calculate the scale
    image_num = 1 + (options->last_img - options->first_img) / options->skip_img;
    float scale = (float) (sqrt(3.0) / (double) image_num);
    scale = scale * options->scale;

    // Load static kernel arguments
    kargs->scale = scale;
    kargs->vol_offset.x = vol->offset[0];
    kargs->vol_offset.y = vol->offset[1];
    kargs->vol_offset.z = vol->offset[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_pix_spacing.x = vol->pix_spacing[0];
    kargs->vol_pix_spacing.y = vol->pix_spacing[1];
    kargs->vol_pix_spacing.z = vol->pix_spacing[2];


    ////// TIMING CODE //////////////////////
    // Initialize Windows HighRes Timer
    int count,a;
    float kernel_total, io_total;
#if defined (_WIN32)
    LARGE_INTEGER ticksPerSecond;
    LARGE_INTEGER tick;   // A point in time
    LARGE_INTEGER start_ticks_kernel, end_ticks_kernel, cputime;   
    LARGE_INTEGER start_ticks_io, end_ticks_io;   
    LARGE_INTEGER start_ticks_total, end_ticks_total;   
#endif

#if defined (VERBOSE)
    printf("\n\nInitializing High Resolution Timers\n");
#endif
    io_total=0;
    kernel_total=0;

    // get the high resolution counter's accuracy
#if defined (_WIN32)
    if (!QueryPerformanceFrequency(&ticksPerSecond))
	printf("QueryPerformance not present!");
#endif

    // Test: Get current time.
#if defined (_WIN32)
    if (!QueryPerformanceCounter(&tick))
	printf("no go counter not installed");  
#endif
	
    //Create DRR directory
#if defined (_WIN32)
    char drr_dir[1024];
    sprintf (drr_dir, "%s\\DRR", options->input_dir);
    CreateDirectory(drr_dir,NULL);
#endif

    //cudaMalloc( (void**)&dev_vol, vol->npix*sizeof(float));
    //cudaMemset( (void *) dev_vol, 0, vol_size_malloc);	
    //   checkCUDAError("Unable to allocate data volume");
    //float *tmp=(float *)malloc(vol->npix*sizeof(float));
    //    memcpy((void *)tmp,(void *)vol->img,vol->npix*sizeof(float));
    //float *vol_img=(float *)vol->img;
    //for(i=0; i<vol->dim[0]; i++)
    //	for(j=0; j<vol->dim[1]; j++)
    //		for(k=0; k<vol->dim[2]; k++)
    //			vol_img[j*vol->dim[0]*vol->dim[2]+k*vol->dim[0]+i]=tmp[k*vol->dim[0]*vol->dim[1]+j*vol->dim[0]+i];
    //free(tmp);

    ////////////////////////////////////////////////////



    // prepare texture
    cudaChannelFormatDesc ca_descriptor;
    cudaExtent ca_extent;
    cudaArray *dev_3Dvol=0;

    ca_descriptor = cudaCreateChannelDesc<float>();
    ca_extent.width  = vol->dim[0];
    ca_extent.height = vol->dim[1];
    ca_extent.depth  = vol->dim[2];
    //ca_extent.width  = vol->dim[0];
    //ca_extent.height = vol->dim[2];
    //ca_extent.depth  = vol->dim[1];
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
    cpy_params.srcPtr   = make_cudaPitchedPtr( (void*)vol->img, ca_extent.width *sizeof(float), ca_extent.width , ca_extent.height  );

    cudaMemcpy3D( &cpy_params );

    // cudaMemcpy(dev_vol,  vol->img, vol->npix * vol->pix_size, cudaMemcpyHostToDevice );


#if defined (VERBOSE)
    printf(" done.\n\n");

#endif

    // This is just to retrieve the 2D image dimensions
    int fimg=options->first_img;
    do{
	cbi = get_image_raw (options, fimg);
	fimg++;
    }
    while(cbi==NULL);
		
    cudaMalloc( (void**)&dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float)); 

    cudaMalloc( (void**)&dev_coef, 7*cbi->dim[0]*cbi->dim[1]*sizeof(float));
    checkCUDAError("Unable to allocate coef devmem");
    host_coef=(float*)malloc(7*cbi->dim[0]*cbi->dim[1]*sizeof(float));
		
    proj_image_free( cbi );

    ////// TIMING CODE //////////////////////
#if defined (_WIN32)
    QueryPerformanceCounter(&start_ticks_total);
#endif
    /////////////////////////////////////////

    printf ("Projecting Image:");
    // Project each image into the volume one at a time
    for(image_num = options->first_img;  image_num <= options->last_img;  image_num += options->skip_img)
    {

	printf(" %d\n",image_num);
	fflush(stdout);

	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&start_ticks_io);
#endif
#endif
	/////////////////////////////////////////


	// Load the current image
	cbi = get_image_raw (options, image_num);
	if (cbi==NULL)
	    continue;

	// Load dynamic kernel arguments
	kargs->img_dim.x = cbi->dim[0];
	kargs->img_dim.y = cbi->dim[1];
	kargs->ic.x = cbi->ic[0];
	kargs->ic.y = cbi->ic[1];
	kargs->nrm.x = cbi->nrm[0];
	kargs->nrm.y = cbi->nrm[1];
	kargs->nrm.z = cbi->nrm[2];
	kargs->sad = cbi->sad;
	kargs->sid = cbi->sid;
	for(i=0; i<12; i++)
	    kargs->matrix[i] = (float)cbi->matrix[i];

	//Precalculate coeff

	int xy7;
	double * ic=cbi->ic;
	for (int x=0;x<cbi->dim[0];x++)
	    for (int y=0; y<cbi->dim[1];y++){
		xy7=7*(y*cbi->dim[0]+x);
		host_coef[xy7]  =((y-ic[1])*cbi->matrix[8]-cbi->matrix[4])/(cbi->matrix[5]-(y-ic[1])*cbi->matrix[9]);
		host_coef[xy7+2]=((y-ic[1])*cbi->matrix[9]-cbi->matrix[5])/(cbi->matrix[4]-(y-ic[1])*cbi->matrix[8]);
		host_coef[xy7+1]=(y-ic[1])*cbi->matrix[11]/(cbi->matrix[5]-(y-ic[1])*cbi->matrix[9]);
		host_coef[xy7+3]=(y-ic[1])*cbi->matrix[11]/(cbi->matrix[4]-(y-ic[1])*cbi->matrix[8]);
		host_coef[xy7+4]=(x-ic[0])*cbi->matrix[8]/cbi->matrix[2];
		host_coef[xy7+5]=(x-ic[0])*cbi->matrix[9]/cbi->matrix[2];
		host_coef[xy7+6]=(x-ic[0])*cbi->matrix[11]/cbi->matrix[2];
	    }

	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&start_ticks_io);
#endif
#endif
	/////////////////////////////////////////

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

	
	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&end_ticks_io);
	cputime.QuadPart = end_ticks_io.QuadPart- start_ticks_io.QuadPart;
	io_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
	QueryPerformanceCounter(&start_ticks_kernel);
#endif
#endif
	/////////////////////////////////////////
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



	////// TIMING CODE //////////////////////
#if defined (TIME_KERNEL)
#if defined (_WIN32)
	QueryPerformanceCounter(&end_ticks_kernel);
	cputime.QuadPart = end_ticks_kernel.QuadPart- start_ticks_kernel.QuadPart;
	kernel_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
#endif
#endif
	/////////////////////////////////////////

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
	proj_image_free( cbi );

    }

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif
	

	

	
    ////// TIMING CODE //////////////////////
    // Report Timing Data
#if defined (_WIN32)
#if defined (TIME_KERNEL)
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf ("\tTotal Kernel  Time: %.9fs\n", kernel_total);
    printf ("\tTotal File IO Time: %.9fs\n\n", io_total);

    printf ("[Average Projection Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart)/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage Kernel  Time: %.9fs\n", kernel_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf ("\tAverage File IO Time: %.9fs\n\n", io_total/ (1+(options->last_img - options->first_img) / options->skip_img));
    printf("========================================\n");
#else
    QueryPerformanceCounter(&end_ticks_total);
    cputime.QuadPart = end_ticks_total.QuadPart- start_ticks_total.QuadPart;
    printf("========================================\n");
    printf ("[Total Execution Time: %.9fs ]\n", ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart));
    printf("========================================\n");
#endif
#endif
    /////////////////////////////////////////


    // Cleanup
    cudaFree( dev_img );
    cudaFree( dev_kargs );
    cudaFree( dev_matrix );
    cudaFree( dev_vol );
    cudaFree( dev_coef);
    free(host_coef);

    return 0;
}
//}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: checkCUDAError() /////////////////////////////////////////////
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
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
