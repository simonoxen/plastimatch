/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

#define _CRT_SECURE_NO_DEPRECATE
#define READ_PFM
#define WRITE_BLOCK (1024*1024)

/****************************************************\
* Uncomment the line below to enable verbose output. *
* Enabling this should not nerf performance.         *
\****************************************************/
//#define VERBOSE

/**********************************************************\
* Uncomment the line below to enable detailed performance  *
* reporting.  This measurement alters the system, however, *
* resulting in significantly slower kernel execution.      *
\**********************************************************/
//#define TIME_KERNEL

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
#include "fdk.h"
#include "fdk_opts.h"
#include "fdk_cuda.h"
#include "volume.h"
#include "mathutil.h"

/*********************
* High Res Win Timer *
*********************/
#include <time.h>
#include <windows.h>
#include <winbase.h>



// P R O T O T Y P E S ////////////////////////////////////////////////////
Volume* my_create_volume (MGHCBCT_Options* options);
CB_Image* load_cb_image (char* img_filename, char* mat_filename);
CB_Image* get_image (MGHCBCT_Options* options, int image_num);
void free_cb_image (CB_Image* cbi);
float convert_to_hu_pixel (float in_value);
void convert_to_hu (Volume* vol, MGHCBCT_Options* options);
void write_mha (char* filename, Volume* vol);
void fwrite_block (void* buf, size_t size, size_t count, FILE* fp);

int CUDA_reconstruct_conebeam (Volume *vol, MGHCBCT_Options *options);
void checkCUDAError(const char *msg);

__global__ void kernel_fdk (float *dev_vol, float *dev_img, float *matrix, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing, unsigned int Blocks_Y, float invBlocks_Y);
///////////////////////////////////////////////////////////////////////////



// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
///////////////////////////////////////////////////////////////////////////



//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
__global__
void kernel_fdk (float *dev_vol, float *dev_img, float *matrix, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_offset, int3 vol_dim, float3 vol_pix_spacing, unsigned int Blocks_Y, float invBlocks_Y)
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

	float4 matrix_x;
	float4 matrix_y;
	float4 matrix_z;

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
	voxel_data = tex1Dfetch(tex_img, ip.x*img_dim.x + ip.y);

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



///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam() //////////////////////////////////
int CUDA_reconstruct_conebeam (Volume *vol, MGHCBCT_Options *options)
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

	// Structure for passing arugments to kernel: (See fdk_cuda.h)
	kernel_args_fdk *kargs;
	kargs = (kernel_args_fdk *) malloc(sizeof(kernel_args_fdk));

	CB_Image* cbi;
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
		LARGE_INTEGER ticksPerSecond;
		LARGE_INTEGER tick;   // A point in time
		LARGE_INTEGER start_ticks_kernel, end_ticks_kernel, cputime;   
		LARGE_INTEGER start_ticks_io, end_ticks_io;   
		LARGE_INTEGER start_ticks_total, end_ticks_total;   

		#if defined (VERBOSE)
			printf("\n\nInitializing High Resolution Timers\n");
		#endif

		// get the high resolution counter's accuracy
		if (!QueryPerformanceFrequency(&ticksPerSecond))
			printf("QueryPerformance not present!");

		#if defined (VERBOSE)
			printf ("\tFreq Test:   %I64Ld ticks/sec\n",ticksPerSecond    );
		#endif

		// Test: Get current time.
		if (!QueryPerformanceCounter(&tick))
			printf("no go counter not installed");  

		#if defined (VERBOSE)
			printf ("\tTestpoint:   %I64Ld  ticks\n",tick);
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
	cbi = get_image(options, options->first_img);
	cudaMalloc( (void**)&dev_img, cbi->dim[0]*cbi->dim[1]*sizeof(float)); 
	free_cb_image( cbi );

		////// TIMING CODE //////////////////////
		QueryPerformanceCounter(&start_ticks_total);
		/////////////////////////////////////////

	// Project each image into the volume one at a time
	for(image_num = options->first_img;  image_num <= options->last_img;  image_num += options->skip_img)
	{
			////// TIMING CODE //////////////////////
			#if defined (TIME_KERNEL)
				QueryPerformanceCounter(&start_ticks_io);
			#endif
			/////////////////////////////////////////

		// Load the current image
		cbi = get_image(options, image_num);

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
		free_cb_image( cbi );

			////// TIMING CODE //////////////////////
			#if defined (TIME_KERNEL)
				QueryPerformanceCounter(&end_ticks_io);
				cputime.QuadPart = end_ticks_io.QuadPart- start_ticks_io.QuadPart;
				io_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
				QueryPerformanceCounter(&start_ticks_kernel);
			#endif
			/////////////////////////////////////////

		// Invoke ze kernel  \(^_^)/
		//-------------------------------------
		kernel_fdk<<< dimGrid, dimBlock >>>(dev_vol,
						    dev_img,
						    dev_matrix,
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
				QueryPerformanceCounter(&end_ticks_kernel);
				cputime.QuadPart = end_ticks_kernel.QuadPart- start_ticks_kernel.QuadPart;
				kernel_total += ((float)cputime.QuadPart/(float)ticksPerSecond.QuadPart);
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
		/////////////////////////////////////////


	// Cleanup
	cudaFree( dev_img );
	cudaFree( dev_kargs );
	cudaFree( dev_matrix );
	cudaFree( dev_vol );	

	return 0;
}
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
// FUNCTION: get_image() //////////////////////////////////////////////////
CB_Image* get_image (MGHCBCT_Options* options, int image_num)
{
	#if defined (READ_PFM)
		char* img_file_pat = "out_%04d.pfm";
	#else
		char* img_file_pat = "out_%04d.pgm";
	#endif
		char* mat_file_pat = "out_%04d.txt";
	
	char img_file[1024], mat_file[1024], fmt[1024];
	sprintf (fmt, "%s/%s", options->input_dir, img_file_pat);
	sprintf (img_file, fmt, image_num);
	sprintf (fmt, "%s/%s", options->input_dir, mat_file_pat);
	sprintf (mat_file, fmt, image_num);
	return load_cb_image (img_file, mat_file);
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: my_create_volume() ///////////////////////////////////////////
Volume* my_create_volume (MGHCBCT_Options* options)
{
	////////////////////////////////////////////////
	// NOTE: struct Volume is defined in volume.h // 
	////////////////////////////////////////////////
	float offset[3];
	float spacing[3];
	float* vol_size = options->vol_size;
	int* resolution = options->resolution;

	spacing[0] = vol_size[0] / resolution[0];
	spacing[1] = vol_size[1] / resolution[1];
	spacing[2] = vol_size[2] / resolution[2];

	// Position in world coords of the upper left (first) voxel in the volume
	offset[0] = -vol_size[0] / 2.0f + spacing[0] / 2.0f;
	offset[1] = -vol_size[1] / 2.0f + spacing[1] / 2.0f;
	offset[2] = -vol_size[2] / 2.0f + spacing[2] / 2.0f;

	// Note: volume_create() is defined in volume.c
	// This populates the struct Volume as follows:
	//    vol.dim[i] 		= resolution[i]
	//    vol.offset[i] 		= offset[i]
	//    vol.pix_spacing[i] 	= spacing[i]
	//    vol.pix_type 		= PT_FLOAT (enumerated type)
	//			other options:	PT_UNDEFINED
	//					PT_UCHAR
	//					PT_SHORT
	//					PT_VF_FLOAT_INTERLEAVED
	//					PT_VF_FLOAT_PLANAR
	return volume_create (resolution, offset, spacing, PT_FLOAT, 0);
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: load_cb_image() //////////////////////////////////////////////
CB_Image* load_cb_image (char* img_filename, char* mat_filename)
{
    int i;
    size_t rc;
    float f;
    FILE* fp;
    char buf[1024];
    CB_Image* cbi;

    /////////////////////////////////////
    // CB Image (Used as return value)
    ////////////////
    cbi = (CB_Image*) malloc (sizeof(CB_Image));

    
    /////////////////////////////////////
    // Open the IMAGE FILE
    ////////////////
    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", img_filename);
	exit (-1);
    }


    /////////////////////////////////////
    // PARSE THE IMAGE FILE
    ////////////////

    // There are two image formats (PFM and PGM)
    //    we have routines for each...
    

    ////////////////////
    // IMAGE TYPE: PFM
    ////////////
#if defined (READ_PFM)

    /***************************************
     * The PFM image format consists of a  *
     * 3 line header followed by the pixel *
     * data.  First we read the header to  *
     * obtain the image resolution then we *
     * read the pixel data.                *
     ***************************************/
    
    // HEADER //////////////////////////////////////////////////////////////////
    // PFM LINE 01: Verify filetype is PFM
    fgets (buf, 1024, fp);
    if (strncmp(buf, "Pf", 2)) { fprintf (stderr, "Couldn't parse file %s as an image [1]\n", img_filename); printf (buf); exit (-1); }

    
    // PFM LINE 02: Read Image Resolution into cbi->dim[0] and cbi->dim[1]
    fgets (buf, 1024, fp);
    if (2 != sscanf (buf, "%d %d", &cbi->dim[0], &cbi->dim[1]))
	    { fprintf (stderr, "Couldn't parse file %s as an image [2]\n", img_filename); exit (-1); }


    // PFM LINE 03: Skip
    fgets (buf, 1024, fp);
    // END HEADER //////////////////////////////////////////////////////////////


    // PIXEL DATA //////////////////////////////////////////////////////////////
    // Allocate memory for the image pixels
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) { fprintf (stderr, "Couldn't malloc memory for input image\n"); exit (-1); }


    // Read image pixels from file into cbi->img
    rc = fread (cbi->img, sizeof(float), cbi->dim[0] * cbi->dim[1], fp);
    if (rc != cbi->dim[0] * cbi->dim[1]) { fprintf (stderr, "Couldn't load raster data for %s\n", img_filename); exit (-1); }
    // END PIXEL DATA //////////////////////////////////////////////////////////

#else
    
    ////////////////////
    // IMAGE TYPE: PGM
    ////////////

    /***************************************
     * The PGM image format consists of a  *
     * 4 line header followed by the pixel *
     * data.  First we read the header to  *
     * obtain the image resolution then we *
     * read the pixel data.                *
     ***************************************/

    // HEADER //////////////////////////////////////////////////////////////////
    // PGM LINE 01: Verify Magic Number = "P2" (ASCII PGM)
    fgets (buf, 1024, fp);
    if (strncmp(buf, "P2", 2)) { fprintf (stderr, "Couldn't parse file %s as an image [1]\n", img_filename); printf (buf); exit (-1); }

    
    // PGM LINE 02: Skip the comment line
    fgets (buf, 1024, fp);
    

    // PGM LINE 03: Read image resolution into cbi->dim[0] and cbi->dim[1]
    fgets (buf, 1024, fp);
    if (2 != sscanf (buf, "%d %d", &cbi->dim[0], &cbi->dim[1])) { fprintf (stderr, "Couldn't parse file %s as an image [2]\n", img_filename); exit (-1); }


    // PGM LINE 04: Contains maximum grayscale value (we skip this line)
    fgets (buf, 1024, fp);
    // END HEADER //////////////////////////////////////////////////////////////



    // PIXEL DATA //////////////////////////////////////////////////////////////
    // Allocate memory for the image pixels
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) { fprintf (stderr, "Couldn't malloc memory for input image\n"); exit (-1); }


    // Read image pixels from file into cbi->img
    for (i = 0; i < cbi->dim[0] * cbi->dim[1]; i++)
    {
	if (1 != fscanf (fp, "%g", &cbi->img[i]))
		{ fprintf (stderr, "Couldn't parse file %s as an image [3,%d]\n", img_filename, i); exit (-1); }
    }
    // END PIXEL DATA //////////////////////////////////////////////////////////
#endif

    // Close File Pointer to IMG FILE
    fclose (fp);



    /////////////////////////////////////
    // Load the PROJECTION MATRIX FILE
    ////////////////
    fp = fopen (mat_filename,"r");
    if (!fp) { fprintf (stderr, "Can't open file %s for read\n", mat_filename); exit (-1); }

    
    /////////////////////////////////////
    // PARSE THE MATRIX FILE
    ////////////////

    // MAT LINES 1-2: Read image center into cbi->ic[0] and cbi->ic[1]
    for (i = 0; i < 2; i++)
    {
	if (1 != fscanf (fp, "%g", &f))
		{ fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", mat_filename, i); exit (-1); }
	cbi->ic[i] = (double) f;
    }


    // MAT LINES 3-14: Load projection matrix into cbi->matrix[0] thru cbi->matrix[11]
    for (i = 0; i < 12; i++)
    {
	if (1 != fscanf (fp, "%g", &f))
		{ fprintf (stderr, "Couldn't parse file %s as a matrix [2,%d]\n", mat_filename, i); exit (-1); }
	cbi->matrix[i] = (double) f;
    }


    // MAT LINE 15: Load SAD into cbi->sad
    if (1 != fscanf (fp, "%g", &f))
    	{ fprintf (stderr, "Couldn't load sad from %s\n", mat_filename); exit (-1); }
    cbi->sad = (double) f;


    // MAT LINE 16: Load SID into cbi->sid
    if (1 != fscanf (fp, "%g", &f)) { fprintf (stderr, "Couldn't load sid from %s\n", mat_filename); exit (-1); }
    cbi->sid = (double) f;


    // MAT LINE 17-20: Load nrm vector
    for (i = 0; i < 3; i++) {
	if (1 != fscanf (fp, "%g", &f))
		{ fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", mat_filename, i); exit (-1); }
	cbi->nrm[i] = (double) f;
    }
    
    // Close File Pointer to IMG FILE
    fclose (fp);

    #if defined (commentout)
	printf ("Image center: ");
	rawvec2_print_eol (stdout, cbi->ic);
	printf ("Projection matrix:\n");
	matrix_print_eol (stdout, cbi->matrix, 3, 4);
    #endif

    return cbi;
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: free_cb_image() //////////////////////////////////////////////
void free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: convert_to_hu_pixel() ////////////////////////////////////////
float convert_to_hu_pixel (float in_value)
{
// Needs work / rewrite
    float hu;
    float diameter = 40.0;  /* reconstruction diameter in cm */
    hu = 1000 * ((in_value / diameter) - .167) / .167;
    return hu;
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: convert_to_hu() //////////////////////////////////////////////
void convert_to_hu (Volume* vol, MGHCBCT_Options* options)
{
    int i, j, k, p;
    float* img = (float*) vol->img;

// Needs work / rewrite
    // hu value of 0 is the density of water
    // hu value of -1000 is density of air
    // hu value of 1000 is density of bone
    
    p = 0;
    for (k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++) {
		img[p] = convert_to_hu_pixel (img[p]);
		p++;
	    }
	}
    }
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: fwrite_block() ///////////////////////////////////////////////
void fwrite_block (void* buf, size_t size, size_t count, FILE* fp)
{
    size_t left_to_write = count * size;
    size_t cur = 0;
    char* bufc = (char*) buf;

    while (left_to_write > 0) {
	size_t this_write, rc;

	this_write = left_to_write;
	if (this_write > WRITE_BLOCK) this_write = WRITE_BLOCK;
	rc = fwrite (&bufc[cur], 1, this_write, fp);
	if (rc != this_write) {
	    fprintf (stderr, "Error writing to file.  rc=%d, this_write=%d\n",
		    rc, this_write);
	    return;
	}
	cur += rc;
	left_to_write -= rc;
    }
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: write_mha() //////////////////////////////////////////////////
void write_mha (char* filename, Volume* vol)
{
    FILE* fp;
    char* mha_header = 
	    "ObjectType = Image\n"
	    "NDims = 3\n"
	    "BinaryData = True\n"
	    "BinaryDataByteOrderMSB = False\n"
	    "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
	    "Offset = %g %g %g\n"
	    "CenterOfRotation = 0 0 0\n"
	    "ElementSpacing = %g %g %g\n"
	    "DimSize = %d %d %d\n"
	    "AnatomicalOrientation = RAI\n"
	    "%s"
	    "ElementType = %s\n"
	    "ElementDataFile = LOCAL\n";

    if (vol->pix_type == PT_VF_FLOAT_PLANAR) {
	fprintf (stderr, "Error, PT_VF_FLOAT_PLANAR not implemented\n");
	exit (-1);
    }

    fp = fopen (filename,"wb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for write\n", filename);
	return;
    }
    fprintf (fp, mha_header, 
	     vol->offset[0], vol->offset[1], vol->offset[2], 
	     vol->pix_spacing[0], vol->pix_spacing[1], vol->pix_spacing[2], 
	     vol->dim[0], vol->dim[1], vol->dim[2],
	     (vol->pix_type == PT_VF_FLOAT_INTERLEAVED) 
	     ? "ElementNumberOfChannels = 3\n" : "",
		 (vol->pix_type == PT_SHORT) ? "MET_SHORT" : (vol->pix_type == PT_UCHAR ? "MET_UCHAR" : "MET_FLOAT"));
    fflush (fp);

    fwrite_block (vol->img, vol->pix_size, vol->npix, fp);

    fclose (fp);
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// FUNCTION: main() ///////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	MGHCBCT_Options options;
	Volume* vol;


	////////////////////////////////////
	#if defined (READ_PFM)
		printf("[PFM Input]\n");
	#else
		printf("[PGM Input]\n"); 
	#endif
	////////////////////////////////////


	/**************************************************************** 
	* STEP 0: Parse commandline arguments                           * 
	****************************************************************/
	parse_args (&options, argc, argv);
	
	/*****************************************************
	* STEP 1: Create the 3D array of voxels              *
	*****************************************************/
	vol = my_create_volume (&options);

	/***********************************************
	* STEP 2: Reconstruct/populate the volume      *
	***********************************************/
	CUDA_reconstruct_conebeam (vol, &options);	

	/*************************************
	* STEP 3: Convert to HU values       *
	*************************************/
	convert_to_hu (vol, &options);

	/*************************************
	* STEP 4: Write MHA output file      *
	*************************************/
	printf("Writing output volume...");
	write_mha (options.output_file, vol);
	printf(" done.\n\n");

	return 0;
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// Vim Editor Settings ////////////////////////////////////////////////////
/* vim:ts=8:sw=8:cindent:nowrap */
///////////////////////////////////////////////////////////////////////////
