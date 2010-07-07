/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <brook/brook.hpp>
//#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "fdk_brook.h"
#include "fdk_opts.h"
#include "fdk_utils.h"
#include "math_util.h"
#include "mha_io.h"
#include "proj_image.h"
#include "proj_image_dir.h"
#include "plm_timer.h"
#include "volume.h"

#include "fdk_brook_kernel.cpp"

#if defined (commentout)
void
fdk_brook_c (Volume* vol, Proj_image_dir* proj_dir, Fdk_options* options)
{
    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float sq_vol_dim;
    float3 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    Proj_image* cbi;
    Proj_matrix *pmat;
	
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
    cbi = get_image_pfm (options, options->first_img);
    pmat = cbi->pmat;
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
	pmat = cbi->pmat;

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	image_preparation_cycles +=  (clock_end - clock_start);

	/* Calculating xip, yip, and zip */
	/* Calculating xip, yip, zip, wip data, ic */
	xip.x = (float) (pmat->matrix[0] + pmat->ic[0] * pmat->matrix[8]);
	xip.y = (float) (pmat->matrix[4] + pmat->ic[1] * pmat->matrix[8]);
	xip.z = (float) (pmat->matrix[8]);

	yip.x = (float) (pmat->matrix[1] + pmat->ic[0] * pmat->matrix[9]);
	yip.y = (float) (pmat->matrix[5] + pmat->ic[1] * pmat->matrix[9]);
	yip.z = (float) (pmat->matrix[9]);

	zip.x = (float) (pmat->matrix[2] + pmat->ic[0] * pmat->matrix[10]);
	zip.y = (float) (pmat->matrix[6] + pmat->ic[1] * pmat->matrix[10]);
	zip.z = (float) (pmat->matrix[10]);
			
	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;

	streamRead(img_s, cbi->img);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	i_cycles +=  (clock_end - clock_start);


	QueryPerformanceCounter(&clock_count);
	clock_start = (double)clock_count.QuadPart;	

	float sad_sid_2 = (pmat->sad * pmat->sad)/(pmat->sid * pmat->sid);
	k_scale(img_s, scale, sad_sid_2, img_s);

	k_comp_xy(xip, spacing.x, offset.x, xip_o);
	k_comp_xy(yip, spacing.y, offset.y, yip_o);

	float3 val = float3(pmat->ic[0] * pmat->matrix[11] + pmat->matrix[3],
			    pmat->ic[1] * pmat->matrix[11] + pmat->matrix[7],
			    pmat->matrix[11]);
	k_comp_z(zip, spacing.z, offset.z, val, zip_o);
		
	// float sad2 = (pmat->sad)*(pmat->sad);
	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
			      (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
			      img_dim, img_s, imgGPU);
				   

	k_sum_volume(volumeGPU, imgGPU, volumeGPU);

	QueryPerformanceCounter(&clock_count);
	clock_end = (double)clock_count.QuadPart;
	processing_cycles +=  (clock_end - clock_start);
			
	proj_image_destroy (cbi);
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
#endif

void
fdk_brook_reconstruct (
    Volume* vol, 
    Proj_image_dir *proj_dir, 
    Fdk_options* options
)
{
    int i;
    int num_imgs;
    float scale;
    Timer timer;
    double backproject_time = 0.0;
    double filter_time = 0.0;
    double io_time = 0.0;

    num_imgs = 1 + (options->last_img - options->first_img)
	/ options->skip_img;

    scale = (float) (sqrt(3.0) / (double) num_imgs);
    scale = scale * options->scale;

    /* Set brook variables */
    float3 xip, yip, zip, spacing, offset, wip;
    float2 img_dim;
    float sq_vol_dim;
    spacing.x = vol->pix_spacing[0];
    spacing.y = vol->pix_spacing[1];
    spacing.z = vol->pix_spacing[2];
    offset.x = vol->offset[0];
    offset.y = vol->offset[1];
    offset.z = vol->offset[2];
    sq_vol_dim = (float) vol->dim[0]*vol->dim[1];
    float* volumeGPU_data = (float*) vol->img;

    /* Compute texture size in X and Y dimensions needed to store volume */
    int size = (int)ceil(sqrt((double)vol->npix/4));
    int texture_size = size*size; // Texture of float4 elements
    printf ("A texture memory of %d x %d is needed for the volume \n", 
	size, size);
    float* temp_vol = (float*)malloc(sizeof(float)*texture_size*4);
    if(temp_vol == NULL){
	printf("Memory allocaton failed.....Exiting\n");
	exit(1);
    }	

    /* Initial Image Dimension and scaling */
    Proj_image* cbi;
    cbi = proj_image_dir_load_image (proj_dir, options->first_img);
    img_dim.x = (float) cbi->dim[1];
    img_dim.y = (float) cbi->dim[0];

    /* Declare the variolus input and output streams */
    ::brook::stream imgGPU(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream volumeGPU(::brook::getStreamType(( float4  *)0), size , size,-1);
    ::brook::stream img_s(::brook::getStreamType(( float  *)0), cbi->dim[1], cbi->dim[0],-1);
    ::brook::stream xip_o(::brook::getStreamType(( float3  *)0), vol->dim[0],-1);
    ::brook::stream yip_o(::brook::getStreamType(( float3  *)0), vol->dim[1],-1);
    ::brook::stream zip_o(::brook::getStreamType(( float3  *)0), vol->dim[2],-1);
    proj_image_destroy (cbi);

    /* Initialize output volume stream with zeros */
    k_assign(volumeGPU);

    for (i = options->first_img; 
	 i <= options->last_img; 
	 i += options->skip_img)
    {
	Proj_image* cbi;
	Proj_matrix *pmat;
	printf ("Processing image %d\n", i);
	plm_timer_start (&timer);
	cbi = proj_image_dir_load_image (proj_dir, i);
	pmat = cbi->pmat;
	io_time += plm_timer_report (&timer);

	if (options->filter == FDK_FILTER_TYPE_RAMP) {
	    plm_timer_start (&timer);
	    proj_image_filter (cbi);
	    filter_time += plm_timer_report (&timer);
	}

	/* Calculating xip, yip, and zip */
	xip.x = (float) (pmat->matrix[0] + pmat->ic[0] * pmat->matrix[8]);
	xip.y = (float) (pmat->matrix[4] + pmat->ic[1] * pmat->matrix[8]);
	xip.z = (float) (pmat->matrix[8]);

	yip.x = (float) (pmat->matrix[1] + pmat->ic[0] * pmat->matrix[9]);
	yip.y = (float) (pmat->matrix[5] + pmat->ic[1] * pmat->matrix[9]);
	yip.z = (float) (pmat->matrix[9]);

	zip.x = (float) (pmat->matrix[2] + pmat->ic[0] * pmat->matrix[10]);
	zip.y = (float) (pmat->matrix[6] + pmat->ic[1] * pmat->matrix[10]);
	zip.z = (float) (pmat->matrix[10]);

	plm_timer_start (&timer);

	/* Load and scale input image */
	float sad_sid_2 = (pmat->sad * pmat->sad)/(pmat->sid * pmat->sid);
	streamRead (img_s, cbi->img);
	k_scale (img_s, scale, sad_sid_2, img_s);

	/* Precompute xip_o, yip_o, and zip_o */
	k_comp_xy(xip, spacing.x, offset.x, xip_o);
	k_comp_xy(yip, spacing.y, offset.y, yip_o);
	float3 val = float3(pmat->ic[0] * pmat->matrix[11] + pmat->matrix[3],
	    pmat->ic[1] * pmat->matrix[11] + pmat->matrix[7],
	    pmat->matrix[11]);
	k_comp_z(zip, spacing.z, offset.z, val, zip_o);

	/* Do backprojection */
	k_get_pixel_optimized(xip_o, yip_o, zip_o, (float)size, 
	    (float)vol->dim[0], (float)vol->dim[1], (float)vol->dim[2],
	    img_dim, img_s, imgGPU);
	k_sum_volume (volumeGPU, imgGPU, volumeGPU);

	backproject_time += plm_timer_report (&timer);

	proj_image_destroy (cbi);
    }

    streamWrite (volumeGPU, temp_vol);
    // Write volume to the vol data structure
    float *temp = (float *)vol->img;
    for (int i = 0; i < vol->npix; i++)
	temp[i] = temp_vol[i];

    free(temp_vol);

    printf ("I/O time = %g\n", io_time);
    printf ("Filter time = %g\n", filter_time);
    printf ("Backprojection time = %g\n", backproject_time);
}
