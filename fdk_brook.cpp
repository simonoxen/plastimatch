/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <brook/brook.hpp>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fdk_brook.h"
#include "fdk_opts.h"
#include "fdk_utils.h"
#include "mathutil.h"
#include "proj_image.h"
#include "readmha.h"
#include "volume.h"

#include "fdk_brook_kernel.cpp"

void
fdk_brook_512_256(Volume* vol, Fdk_options* options)
{
    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float3 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    Proj_image* cbi;
	
    /*******************
	Variable Assignment
    ********************/

    /* Set voxel spacing */
    spacing.x = vol->pix_spacing[0]; 
    spacing.y = vol->pix_spacing[1]; 
    spacing.z = vol->pix_spacing[2];

    offset.x = vol->offset[0]; 
    offset.y = vol->offset[1]; 
    offset.z = vol->offset[2];


    // Obtain the texture size in the X and Y dimensions needed to store the volume
//    size = 2048;
    size = 1024;

    /* Initial Image Dimension and scaling */
    cbi = get_image_pfm (options,options->first_img);
    int num_imgs = 1 + (options->last_img - options->first_img)/ options->skip_img;
    scale = (float) (sqrt((double)3) / (double) num_imgs);
    scale = scale * options->scale;
    img_dim.x = (float) cbi->dim[1];
    img_dim.y = (float) cbi->dim[0];


    /* Declare the variolus input and output streams */
    ::brook::stream img_s(::brook::getStreamType(( float  *)0), cbi->dim[1], cbi->dim[0],-1);
    ::brook::stream img_on_gpu(::brook::getStreamType(( float  *)0), cbi->dim[1], cbi->dim[0],-1);
    ::brook::stream xip_o(::brook::getStreamType(( float3  *)0), vol->dim[0],-1);
    ::brook::stream yip_o(::brook::getStreamType(( float3  *)0), vol->dim[1],-1);
    ::brook::stream zip_o(::brook::getStreamType(( float3  *)0), vol->dim[2],-1);

    ::brook::stream imgGPU_0(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream imgGPU_1(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream imgGPU_2(::brook::getStreamType(( float4  *)0), size , size,-1);
    // ::brook::stream imgGPU_3(::brook::getStreamType(( float4  *)0), size , size,-1);

    ::brook::stream volumeGPU_0(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream volumeGPU_1(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream volumeGPU_2(::brook::getStreamType(( float4  *)0), size , size,-1);
    // ::brook::stream volumeGPU_3(::brook::getStreamType(( float4  *)0), size , size,-1);
	
    /* Timing-related stuff */
    double start_time, end_time;

    LARGE_INTEGER clock_count;
    LARGE_INTEGER clock_frequency;
    double clock_start, clock_end;
    double i_cycles = 0.0;
    double o_cycles = 0.0;
    double processing_cycles = 0.0;
    double image_preparation_cycles = 0.0;

    QueryPerformanceFrequency(&clock_frequency);
	 
    QueryPerformanceCounter(&clock_count);
    start_time = (double)clock_count.QuadPart/(double)clock_frequency.QuadPart;
	
    k_assign(volumeGPU_0); // Initialize volume stream with zeros
    k_assign(volumeGPU_1); 
    k_assign(volumeGPU_2); 
    // k_assign(volumeGPU_3); 

    for (image_num = options->first_img; image_num <= options->last_img; image_num += options->skip_img){
	// printf("Projecting image %d\n",image_num);

	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;
		
	cbi = get_image_pfm(options,image_num);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	image_preparation_cycles +=  (clock_end - clock_start);

	/* Calculating xip, yip, and zip */
	/* Calculating xip, yip, zip, wip data, ic */
	xip.x = (float) (cbi->matrix[0] + cbi->ic[0] * cbi->matrix[8]);
	xip.y = (float) (cbi->matrix[4] + cbi->ic[1] * cbi->matrix[8]);
	xip.z = (float) (cbi->matrix[8]);

	yip.x = (float) (cbi->matrix[1] + cbi->ic[0] * cbi->matrix[9]);
	yip.y = (float) (cbi->matrix[5] + cbi->ic[1] * cbi->matrix[9]);
	yip.z = (float) (cbi->matrix[9]);

	zip.x = (float) (cbi->matrix[2] + cbi->ic[0] * cbi->matrix[10]);
	zip.y = (float) (cbi->matrix[6] + cbi->ic[1] * cbi->matrix[10]);
	zip.z = (float) (cbi->matrix[10]);
			
	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;

	streamRead(img_s, cbi->img);
	k_init_img(img_s, img_on_gpu);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	i_cycles +=  (clock_end - clock_start);

	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;	

	float sad_sid_2 = (cbi->sad * cbi->sad)/(cbi->sid * cbi->sid);
	k_scale(img_s, scale, sad_sid_2, img_s);

	k_comp_xy(xip, spacing.x, offset.x, xip_o);
	k_comp_xy(yip, spacing.y, offset.y, yip_o);

	float3 val = float3(cbi->ic[0] * cbi->matrix[11] + cbi->matrix[3],
			    cbi->ic[1] * cbi->matrix[11] + cbi->matrix[7],
			    cbi->matrix[11]);
	k_comp_z(zip, spacing.z, offset.z, val, zip_o);
		
	// float sad2 = (cbi->sad)*(cbi->sad);
	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
			      (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
			      img_dim, img_on_gpu, imgGPU_0);

	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
			      (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
			      img_dim, img_on_gpu, imgGPU_1);

	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
			      (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
			      img_dim, img_on_gpu, imgGPU_2);
	/*

	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
	(float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
	img_dim, img_s, imgGPU_3);
	*/
				   

	k_sum_volume(volumeGPU_0, imgGPU_0, volumeGPU_0);
	k_sum_volume(volumeGPU_1, imgGPU_1, volumeGPU_1);
	k_sum_volume(volumeGPU_2, imgGPU_2, volumeGPU_2);
	// k_sum_volume(volumeGPU_3, imgGPU_3, volumeGPU_3);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	processing_cycles +=  (clock_end - clock_start);
			
	proj_image_free(cbi);
    } 
		
    int texture_size = size*size; // Texture of float4 elements
    float *sub_volume[4];
    for(int i = 0; i < 1; i++){
	sub_volume[i] = (float*)malloc(sizeof(float)*texture_size*4);
	if(sub_volume[i] == NULL){
	    printf("Memory allocaton failed.....Exiting\n");
	    exit(1);
	}
    }

    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;
    // streamWrite(volumeGPU, vol->img);

    streamWrite(volumeGPU_0, sub_volume[0]);
    streamWrite(volumeGPU_1, sub_volume[0]);
    streamWrite(volumeGPU_2, sub_volume[0]);
    // streamWrite(volumeGPU_3, sub_volume[0]);

    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    o_cycles +=  (clock_end - clock_start);

    QueryPerformanceCounter(&clock_count);
    end_time = (double)clock_count.QuadPart/(double)clock_frequency.QuadPart;
	
    printf("Total time = %f\n", end_time - start_time);
    printf("Time needed to send images = %f\n", i_cycles/(double)clock_frequency.QuadPart);
    printf("Time needed to read back volume = %f\n", o_cycles/(double)clock_frequency.QuadPart);
    printf("Processing time on GPU = %f\n", processing_cycles/(double)clock_frequency.QuadPart);
    printf("Time needed to prepare images = %f\n", image_preparation_cycles/(double)clock_frequency.QuadPart);

    /* Write volume to the vol data structure
       float *temp = (float *)vol->img;
       for(int i = 0; i < vol->npix; i++)
       temp[i] = temp_vol[i];

       free(temp_vol);
    */
}

void
fdk_brook_c (Volume* vol, Fdk_options* options)
{
    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float sq_vol_dim;
    float3 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    Proj_image* cbi;
	
    /*******************
	Variable Assignment
    ********************/

    /* Set voxel spacing */
    spacing.x = vol->pix_spacing[0]; 
    spacing.y = vol->pix_spacing[1]; 
    spacing.z = vol->pix_spacing[2];

    offset.x = vol->offset[0]; 
    offset.y = vol->offset[1]; 
    offset.z = vol->offset[2];

    sq_vol_dim = (float) vol->dim[0]*vol->dim[1];

    // Obtain the texture size in the X and Y dimensions needed to store the volume
    size = (int)ceil(sqrt((double)vol->npix/4));
    printf("A texture memory of %d x %d is needed to accommodate the volume \n", size, size);

    int texture_size = size*size; // Texture of float4 elements
    float* temp_vol = (float*)malloc(sizeof(float)*texture_size*4);
    if(temp_vol == NULL){
	printf("Memory allocaton failed.....Exiting\n");
	exit(1);
    }	

    /* Initial Image Dimension and scaling */
    cbi = get_image_pfm(options,options->first_img);
    int num_imgs = 1 + (options->last_img - options->first_img)/ options->skip_img;
    scale = (float) (sqrt((double)3) / (double) num_imgs);
    scale = scale * options->scale;
    img_dim.x = (float) cbi->dim[1];
    img_dim.y = (float) cbi->dim[0];

    /* Declare the variolus input and output streams */
    ::brook::stream imgGPU(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream volumeGPU(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream img_s(::brook::getStreamType(( float  *)0), cbi->dim[1], cbi->dim[0],-1);
    ::brook::stream xip_o(::brook::getStreamType(( float3  *)0), vol->dim[0],-1);
    ::brook::stream yip_o(::brook::getStreamType(( float3  *)0), vol->dim[1],-1);
    ::brook::stream zip_o(::brook::getStreamType(( float3  *)0), vol->dim[2],-1);
	
    /* Timing-related stuff */
    double start_time, end_time;

    LARGE_INTEGER clock_count;
    LARGE_INTEGER clock_frequency;
    double clock_start, clock_end;
    double i_cycles = 0.0;
    double o_cycles = 0.0;
    double processing_cycles = 0.0;
    double image_preparation_cycles = 0.0;

    QueryPerformanceFrequency(&clock_frequency);
	 
    QueryPerformanceCounter(&clock_count);
    start_time = (double)clock_count.QuadPart/(double)clock_frequency.QuadPart;
	
    k_assign(volumeGPU); // Initialize volume stream with zeros

    for (image_num = options->first_img; image_num <= options->last_img; image_num += options->skip_img){
	// printf("Projecting image %d\n",image_num);

	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;
		
	cbi = get_image_pfm(options,image_num);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	image_preparation_cycles +=  (clock_end - clock_start);

	/* Calculating xip, yip, and zip */
	/* Calculating xip, yip, zip, wip data, ic */
	xip.x = (float) (cbi->matrix[0] + cbi->ic[0] * cbi->matrix[8]);
	xip.y = (float) (cbi->matrix[4] + cbi->ic[1] * cbi->matrix[8]);
	xip.z = (float) (cbi->matrix[8]);

	yip.x = (float) (cbi->matrix[1] + cbi->ic[0] * cbi->matrix[9]);
	yip.y = (float) (cbi->matrix[5] + cbi->ic[1] * cbi->matrix[9]);
	yip.z = (float) (cbi->matrix[9]);

	zip.x = (float) (cbi->matrix[2] + cbi->ic[0] * cbi->matrix[10]);
	zip.y = (float) (cbi->matrix[6] + cbi->ic[1] * cbi->matrix[10]);
	zip.z = (float) (cbi->matrix[10]);
			
	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;

	streamRead(img_s, cbi->img);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	i_cycles +=  (clock_end - clock_start);


	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;	

	float sad_sid_2 = (cbi->sad * cbi->sad)/(cbi->sid * cbi->sid);
	k_scale(img_s, scale, sad_sid_2, img_s);

	k_comp_xy(xip, spacing.x, offset.x, xip_o);
	k_comp_xy(yip, spacing.y, offset.y, yip_o);

	float3 val = float3(cbi->ic[0] * cbi->matrix[11] + cbi->matrix[3],
			    cbi->ic[1] * cbi->matrix[11] + cbi->matrix[7],
			    cbi->matrix[11]);
	k_comp_z(zip, spacing.z, offset.z, val, zip_o);
		
	// float sad2 = (cbi->sad)*(cbi->sad);
	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
			      (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
			      img_dim, img_s, imgGPU);
				   

	k_sum_volume(volumeGPU, imgGPU, volumeGPU);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	processing_cycles +=  (clock_end - clock_start);
			
	proj_image_free(cbi);
    } 
		
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

    // streamWrite(volumeGPU, vol->img);
    streamWrite(volumeGPU, temp_vol);

    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    o_cycles +=  (clock_end - clock_start);

    QueryPerformanceCounter(&clock_count);
    end_time = (double)clock_count.QuadPart/(double)clock_frequency.QuadPart;
	
    printf("Total time = %f\n", end_time - start_time);
    printf("Time needed to send images = %f\n", i_cycles/(double)clock_frequency.QuadPart);
    printf("Time needed to read back volume = %f\n", o_cycles/(double)clock_frequency.QuadPart);
    printf("Processing time on GPU = %f\n", processing_cycles/(double)clock_frequency.QuadPart);
    printf("Time needed to prepare images = %f\n", image_preparation_cycles/(double)clock_frequency.QuadPart);

    // Write volume to the vol data structure
    float *temp = (float *)vol->img;
    for(int i = 0; i < vol->npix; i++)
	temp[i] = temp_vol[i];

    free(temp_vol);
}

void
fdk_brook_b(Volume* vol, Fdk_options* options)
{
    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float sq_vol_dim;
    float2 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    Proj_image* cbi;
	
    /*******************
	Variable Assignment
    ********************/

    /* Set voxel spacing */
    spacing.x = vol->pix_spacing[0]; 
    spacing.y = vol->pix_spacing[1]; 
    spacing.z = vol->pix_spacing[2];

    offset.x = vol->offset[0]; 
    offset.y = vol->offset[1]; 
    offset.z = vol->offset[2];

    sq_vol_dim = (float) vol->dim[0]*vol->dim[1];

    // Obtain the texture size in the X and Y dimensions needed to store the volume
    size = (int)ceil(sqrt((double)vol->npix/4));
    printf("A texture memory of %d x %d is needed to accommodate the volume \n", size, size);

    int texture_size = size*size; // Texture of float4 elements
    float* temp_vol = (float*)malloc(sizeof(float)*texture_size*4);
    if(temp_vol == NULL){
	printf("Memory allocaton failed.....Exiting\n");
	exit(1);
    }	

    /* Initial Image Dimension and scaling */
    cbi = get_image_pfm(options,options->first_img);
    int num_imgs = 1 + (options->last_img - options->first_img)/ options->skip_img;
    scale = (float) (sqrt((double)3) / (double) num_imgs);
    scale = scale * options->scale;
    img_dim.x = (float) cbi->dim[1];
    img_dim.y = (float) cbi->dim[0];

    /* Declare the variolus input and output streams */
    ::brook::stream imgGPU(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream volumeGPU(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream img_s(::brook::getStreamType(( float  *)0), cbi->dim[1], cbi->dim[0],-1);
    ::brook::stream xip_o(::brook::getStreamType(( float3  *)0), vol->dim[0],-1);
    ::brook::stream yip_o(::brook::getStreamType(( float3  *)0), vol->dim[1],-1);
    ::brook::stream zip_o(::brook::getStreamType(( float3  *)0), vol->dim[2],-1);
    ::brook::stream x(::brook::getStreamType(( float3  *)0), vol->dim[0],-1);
    ::brook::stream y(::brook::getStreamType(( float3  *)0), vol->dim[1],-1);
    ::brook::stream z(::brook::getStreamType(( float3  *)0), vol->dim[2],-1);
	
    /* Timing-related stuff */
    double start_time, end_time;

    LARGE_INTEGER clock_count;
    LARGE_INTEGER clock_frequency;
    double clock_start, clock_end;
    double io_cycles = 0.0;
    double processing_cycles = 0.0;
    double image_preparation_cycles = 0.0;

    QueryPerformanceFrequency(&clock_frequency);
	 
    QueryPerformanceCounter(&clock_count);
    start_time = (double)clock_count.QuadPart/(double)clock_frequency.QuadPart;
	
    k_assign(volumeGPU); // Initialize volume stream with zeros

    for (image_num = options->first_img; image_num <= options->last_img; image_num += options->skip_img){
	// printf("Projecting image %d\n",image_num);

	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;
		
	cbi = get_image_pfm(options,image_num);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	image_preparation_cycles +=  (clock_end - clock_start);

	/* Calculating xip, yip, zip, wip data, ic */
	xip.x = (float) (cbi->matrix[0]);
	xip.y = (float) (cbi->matrix[4]);
	xip.z = (float) (cbi->matrix[8]);

	yip.x = (float) (cbi->matrix[1]);
	yip.y = (float) (cbi->matrix[5]);
	yip.z = (float) (cbi->matrix[9]);

	zip.x = (float) (cbi->matrix[2]);
	zip.y = (float) (cbi->matrix[6]);
	zip.z = (float) (cbi->matrix[10]);

	wip.x = (float) (cbi->matrix[3]);
	wip.y = (float) (cbi->matrix[7]);
	wip.z = (float) (cbi->matrix[11]);

	ic.x = (float)cbi->ic[0];
	ic.y = (float)cbi->ic[1];

	float3 nrm = float3(cbi->nrm[0],cbi->nrm[1],cbi->nrm[2]);
	float sad2 = (cbi->sad)*(cbi->sad);
			
	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;

	streamRead(img_s, cbi->img);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	io_cycles +=  (clock_end - clock_start);


	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;	

	float sad_sid_2 = (cbi->sad * cbi->sad)/(cbi->sid * cbi->sid);
	k_scale(img_s, scale, sad_sid_2, img_s);

	k_comp_xyz(xip, spacing.x, offset.x, nrm.x, 0.0, 0.0, xip_o, x);
	k_comp_xyz(yip, spacing.y, offset.y, nrm.y, 0.0, 0.0, yip_o, y);
	k_comp_xyz(zip, spacing.z, offset.z, nrm.z, 1.0, cbi->sad, zip_o, z);
			
	k_get_pixel(xip_o, yip_o, zip_o, wip, 
		    (float)size, 
		    (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
		    ic, img_dim, img_s, 0, sad2, x, y, z, imgGPU);

	k_sum_volume(volumeGPU, imgGPU, volumeGPU);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	processing_cycles +=  (clock_end - clock_start);
			
	proj_image_free(cbi);
    } 
		
    QueryPerformanceCounter(&clock_count);
    clock_start = (double)clock_count.QuadPart;

    // streamWrite(volumeGPU, vol->img);
    streamWrite(volumeGPU, temp_vol);

    QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    io_cycles +=  (clock_end - clock_start);

    QueryPerformanceCounter(&clock_count);
    end_time = (double)clock_count.QuadPart/(double)clock_frequency.QuadPart;
	
    printf("Total time = %f\n", end_time - start_time);
    printf("I/O time = %f\n", io_cycles/(double)clock_frequency.QuadPart);
    printf("Processing time on GPU = %f\n", processing_cycles/(double)clock_frequency.QuadPart);
    printf("Time needed to prepare images = %f\n", image_preparation_cycles/(double)clock_frequency.QuadPart);

    // Write volume to the vol data structure
    float *temp = (float *)vol->img;
    for(int i = 0; i < vol->npix; i++)
	temp[i] = temp_vol[i];

    free(temp_vol);
}
