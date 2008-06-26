/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <brook/brook.hpp>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "config.h"
#include "mathutil.h"
#include "volume.h"
#include "readmha.h"
#include "fdk.h"
#include "fdk_opts.h"
#include "fdk_brook_kernel.cpp"

#define ROUND_INT(x) ((x > 0) ? (int)(x+0.5) : (int)(ceil(x-0.5)))
#define READ_PFM


Volume*
create_volume (int* dim, float* offset, float* pix_spacing, 
	       enum Pixel_Type pix_type)
{
    int i;
    Volume* vol = (Volume*) malloc (sizeof(Volume));

    for (i = 0; i < 3; i++) {
	vol->dim[i] = dim[i];
	vol->offset[i] = offset[i];
	vol->pix_spacing[i] = pix_spacing[i];
    }
    vol->npix = vol->dim[0] * vol->dim[1] * vol->dim[2];
    vol->pix_type = pix_type;

    switch (pix_type) {
    case PT_SHORT:
	vol->img = (void*) malloc (sizeof(short) * vol->npix);
	memset (vol->img, 0, sizeof(short) * vol->npix);
	break;
    case PT_FLOAT:
	vol->img = malloc (sizeof(float) * vol->npix);
	memset (vol->img, 0, sizeof(float) * vol->npix);
	break;
    }

    /* Compute some auxiliary variables */
    vol->xmin = vol->offset[0] - vol->pix_spacing[0] / 2;
    vol->xmax = vol->xmin + vol->pix_spacing[0] * vol->dim[0];
    vol->ymin = vol->offset[1] - vol->pix_spacing[1] / 2;
    vol->ymax = vol->ymin + vol->pix_spacing[1] * vol->dim[1];
    vol->zmin = vol->offset[2] - vol->pix_spacing[2] / 2;
    vol->zmax = vol->zmin + vol->pix_spacing[2] * vol->dim[2];

    return vol;
}

Volume*
my_create_volume (MGHCBCT_Options* options)
{
    float offset[3];
    float spacing[3];
    float* vol_size = options->vol_size;
    int* resolution = options->resolution;

    spacing[0] = vol_size[0] / resolution[0];
    spacing[1] = vol_size[1] / resolution[1];
    spacing[2] = vol_size[2] / resolution[2];

    offset[0] = -vol_size[0] / 2.0f + spacing[0] / 2.0f;
    offset[1] = -vol_size[1] / 2.0f + spacing[1] / 2.0f;
    offset[2] = -vol_size[2] / 2.0f + spacing[2] / 2.0f;

    return create_volume (resolution, offset, spacing, PT_FLOAT);
}

CB_Image* 
load_cb_image (char* img_filename, char* mat_filename)
{
    int i;
    size_t rc;
    float f;
    FILE* fp;
    char buf[1024];
    CB_Image* cbi;

    cbi = (CB_Image*) malloc (sizeof(CB_Image));

    fp = fopen (img_filename,"rb");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", img_filename);
	exit (-1);
    }

#if defined (READ_PFM)
    /* Verify that it is pfm */
    fgets (buf, 1024, fp);
    if (strncmp(buf, "Pf", 2)) {
	fprintf (stderr, "Couldn't parse file %s as an image [1]\n",
		 img_filename);
	printf (buf);
	exit (-1);
    }
    /* Get image resolution */
    fgets (buf, 1024, fp);
    if (2 != sscanf (buf, "%d %d", &cbi->dim[0], &cbi->dim[1])) {
	fprintf (stderr, "Couldn't parse file %s as an image [2]\n", 
		 img_filename);
	exit (-1);
    }
    /* Skip third line */
    fgets (buf, 1024, fp);

    /* Malloc memory */
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    rc = fread (cbi->img, sizeof(float), cbi->dim[0] * cbi->dim[1], fp);
    if (rc != cbi->dim[0] * cbi->dim[1]) {
	fprintf (stderr, "Couldn't load raster data for %s\n",
		 img_filename);
	exit (-1);
    }
#else
    /* Verify that it is pgm */
    fgets (buf, 1024, fp);
    if (strncmp(buf, "P2", 2)) {
	fprintf (stderr, "Couldn't parse file %s as an image [1]\n",
		 img_filename);
	printf (buf);
	exit (-1);
    }
    /* Skip comment line */
    fgets (buf, 1024, fp);
    /* Get image resolution */
    fgets (buf, 1024, fp);
    if (2 != sscanf (buf, "%d %d", &cbi->dim[0], &cbi->dim[1])) {
	fprintf (stderr, "Couldn't parse file %s as an image [2]\n", 
		 img_filename);
	exit (-1);
    }
    /* Skip max gray */
    fgets (buf, 1024, fp);

    /* Malloc memory */
    cbi->img = (float*) malloc (sizeof(float) * cbi->dim[0] * cbi->dim[1]);
    if (!cbi->img) {
	fprintf (stderr, "Couldn't malloc memory for input image\n");
	exit (-1);
    }

    /* Load pixels */
    for (i = 0; i < cbi->dim[0] * cbi->dim[1]; i++) {
	if (1 != fscanf (fp, "%g", &cbi->img[i])) {
	    fprintf (stderr, "Couldn't parse file %s as an image [3,%d]\n", 
		     img_filename, i);
	    exit (-1);
	}
    }
#endif
    fclose (fp);

    /* Load projection matrix */
    fp = fopen (mat_filename,"r");
    if (!fp) {
	fprintf (stderr, "Can't open file %s for read\n", mat_filename);
	exit (-1);
    }
    /* Load image center */
    for (i = 0; i < 2; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	cbi->ic[i] = (double) f;
    }
    /* Load projection matrix */
    for (i = 0; i < 12; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [2,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	cbi->matrix[i] = (double) f;
    }
    /* Load sad */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n");
	exit (-1);
    }
    cbi->sad = (double) f;
    /* Load sid */
    if (1 != fscanf (fp, "%g", &f)) {
	fprintf (stderr, "Couldn't load sad from %s\n");
	exit (-1);
    }
    cbi->sid = (double) f;
    /* Load nrm vector */
    for (i = 0; i < 3; i++) {
	if (1 != fscanf (fp, "%g", &f)) {
	    fprintf (stderr, "Couldn't parse file %s as a matrix [1,%d]\n", 
		     mat_filename, i);
	    exit (-1);
	}
	cbi->nrm[i] = (double) f;
    }
    fclose (fp);

#if defined (commentout)
    printf ("Image center: ");
    rawvec2_print_eol (stdout, cbi->ic);
    printf ("Projection matrix:\n");
    matrix_print_eol (stdout, cbi->matrix, 3, 4);
#endif

    return cbi;
}

void
free_cb_image (CB_Image* cbi)
{
    free (cbi->img);
    free (cbi);
}

/* get_pixel_value_c seems to be no faster than get_pixel_value_b, 
   despite having two fewer compares. */
inline float
get_pixel_value_c (CB_Image* cbi, double r, double c)
{
    int rr, cc;

    r += 0.5;
    if (r < 0) return 0.0;
    rr = (int) r;
    if (rr >= cbi->dim[1]) return 0.0;

    c += 0.5;
    if (c < 0) return 0.0;
    cc = (int) c;
    if (cc >= cbi->dim[0]) return 0.0;

    return cbi->img[rr*cbi->dim[0] + cc];
}

float
convert_to_hu_pixel (float in_value)
{
    float hu;
    float diameter = 40.0;  /* reconstruction diameter in cm */
    hu = (float)(1000 * ((in_value / diameter) - .167) / .167);
    return hu;
}

void
convert_to_hu (Volume* vol, MGHCBCT_Options* options)
{
    int i, j, k, p;
    float* img = (float*) vol->img;
    
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


CB_Image*
get_image (MGHCBCT_Options* options, int image_num)
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

void
gpu_main_512_256(Volume* vol, MGHCBCT_Options* options){

    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float3 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    CB_Image* cbi;
	
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
    cbi = get_image(options,options->first_img);
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
		
	cbi = get_image(options,image_num);

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
	/* GCS: ??? 
	   k_init_img(img_s, i
	   mg_on_gpu);
	*/
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
			
	free_cb_image(cbi);
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
gpu_main_c(Volume* vol, MGHCBCT_Options* options)
{
    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float sq_vol_dim;
    float3 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    CB_Image* cbi;
	
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
    cbi = get_image(options,options->first_img);
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
		
	cbi = get_image(options,image_num);

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
			
	free_cb_image(cbi);
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
gpu_main_b(Volume* vol, MGHCBCT_Options* options)
{
    float3 xip, yip, zip, spacing, offset, wip;

    int image_num, size;
    float sq_vol_dim;
    float2 ic;
    float2 img_dim;
    float scale;

    float* volumeGPU_data = (float*) vol->img;
    CB_Image* cbi;
	
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
    cbi = get_image(options,options->first_img);
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
		
	cbi = get_image(options,image_num);

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
			
	free_cb_image(cbi);
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


int main(int argc, char* argv[])
{
    MGHCBCT_Options options;
    Volume* vol;
    clock_t start_run;
    clock_t end_run;
    double diff_run;
    
    parse_args (&options, argc, argv);

    vol = my_create_volume (&options);

    start_run = clock();

    gpu_main_c(vol, &options);

    end_run = clock();
    diff_run = double(end_run - start_run)/CLOCKS_PER_SEC;
    printf("Time needed to reconstruct volume = %f\n",diff_run);

    convert_to_hu (vol, &options);
    printf ("Writing output volume\n");
    write_mha (options.output_file, vol);
    return 0;
}
