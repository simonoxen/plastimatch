/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    REFS:
    http://en.wikipedia.org/wiki/B-spline
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
    http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    ----------------------------------------------------------------------- */
/* #include <brook/brook.hpp> */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_brook.h"
#include "toy_kernels.cpp"
#include "bspline_brook_kernel.cpp"

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

int iter = 0;

extern "C" {

void 
check_values(FILE *fp, float cpu_value, float gpu_value, int vox_on_cpu, int vox_on_gpu){
    // fprintf(fp, "%f, %f, %f \n", cpu_value, gpu_value, abs(cpu_value - gpu_value));
	// fprintf(fp, "%6.3f, %6.3f \n", cpu_value, gpu_value);
	
	if(fabs(cpu_value - gpu_value) > 0)
		fprintf(fp, "%f %f \n", cpu_value, gpu_value);
    }

void 
bspline_initialize_streams_on_gpu(Volume *fixed, Volume *moving, Volume *moving_grad, BSPLINE_Parms *parms){
    BSPLINE_Data* bspd = &parms->bspd;

    /* Allocate memory for GPU-specific data structure within the parms data structure */
    parms->data_on_gpu = (BSPLINE_DATA_ON_GPU *)malloc(sizeof(BSPLINE_DATA_ON_GPU));
    BSPLINE_DATA_ON_GPU *data_on_gpu = (BSPLINE_DATA_ON_GPU *)&parms->data_on_gpu;

    /* Compute the size of the texture to hold the moving and static images. The texture on the GPU comprises of float4 elements. */
    int volume_texture_size = (int)ceil(sqrt((double)fixed->npix/4)); // Size of the texture to allocate
    printf("Allocating 2D texture of size %d to store volume information. \n", volume_texture_size);
    data_on_gpu->volume_texture_size = volume_texture_size;

    float* temp_fixed_image = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
    float* temp_moving_image = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
    if((temp_fixed_image == NULL) || (temp_moving_image == NULL)){
		printf("Couldn't allocate texture memory for volume...Exiting. \n");
		exit(-1);
    }
    /* Copy the fixed image */
    memset(temp_fixed_image, 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
    memcpy(temp_fixed_image, fixed->img, sizeof(float)*fixed->npix);

    /* Copy the moving image */
    memset(temp_moving_image, 0, sizeof(float)*volume_texture_size*volume_texture_size*4); 
    memcpy(temp_moving_image, moving->img, sizeof(float)*moving->npix);

    printf("Transferrring the static and moving images to the GPU. \n");
    data_on_gpu->fixed_image_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
    data_on_gpu->moving_image_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
    streamRead(*(data_on_gpu->fixed_image_stream), temp_fixed_image);
    streamRead(*(data_on_gpu->moving_image_stream), temp_moving_image);
   
    free((void *)temp_fixed_image);
    free((void *)temp_moving_image);

	/* Copy the gradient values of the moving image to the GPU */
	float* temp_moving_grad[3];
	int i, m;
	for(i = 0; i < 3; i++){
		temp_moving_grad[i] = (float *)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
		if(temp_moving_grad[i] == NULL){
			printf("Couldn't allocate memory for gradient values...Exiting. \n");
			exit(-1);
		}
		memset(temp_moving_grad[i], 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
	}

	/* Separate the interleaved gradient values into three different arrays */
	m = 0;
	float* img = (float *)moving_grad->img;
	for(i = 0; i < 3*moving_grad->npix; i = i + 3){
		temp_moving_grad[0][m] = img[i];
		temp_moving_grad[1][m] = img[i+1];
		temp_moving_grad[2][m] = img[i+2];
		m++;
	}
	
    /* Copy the gradient values to the GPU*/
	printf("Transferrring gradient values to the GPU. \n");
    data_on_gpu->moving_grad_stream_x = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
	data_on_gpu->moving_grad_stream_y = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
	data_on_gpu->moving_grad_stream_z = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);

    streamRead(*(data_on_gpu->moving_grad_stream_x), temp_moving_grad[0]);
	streamRead(*(data_on_gpu->moving_grad_stream_y), temp_moving_grad[1]);
	streamRead(*(data_on_gpu->moving_grad_stream_z), temp_moving_grad[2]);

	for(i = 0; i < 3; i++) 
		free((void *)temp_moving_grad[i]);

    /* Allocate GPU stream to hold the c_lut data structure */
    int num_regions = bspd->rdims[0]*bspd->rdims[1]*bspd->rdims[2];
    int c_lut_texture_size = (int)ceil(sqrt((double)num_regions*64));
    printf("Allocating 2D texture of size %d x %d to store c_lut information. \n", c_lut_texture_size, c_lut_texture_size);
    data_on_gpu->c_lut_texture_size = c_lut_texture_size;

    float *temp_memory = (float*)malloc(sizeof(float)*c_lut_texture_size*c_lut_texture_size);
    if(!temp_memory){
		printf("Couldn't allocate memory for c_lut texture...Exiting\n");
		exit(-1);
    }
    memset(temp_memory, 0, sizeof(float)*c_lut_texture_size*c_lut_texture_size); 
    /* Note: c_lut entries are integers whereas we need floats for the GPU */
    for(int i = 0; i < num_regions*64; i++) 
		temp_memory[i] = (float)bspd->c_lut[i];

    printf("Transferrring the c_lut texture to the GPU. \n");
    data_on_gpu->c_lut_stream = new ::brook::stream(::brook::getStreamType((float *)0), c_lut_texture_size, c_lut_texture_size, -1);
    streamRead(*(data_on_gpu->c_lut_stream), temp_memory);
    free((void *)temp_memory);

    /* Allocate GPU stream to hold the q_lut data structure */
    int num_voxels_per_region = parms->vox_per_rgn[0]*parms->vox_per_rgn[1]*parms->vox_per_rgn[2]; 
    int q_lut_texture_size = (int)ceil(sqrt((double)num_voxels_per_region*64));
    printf("Allocating 2D texture of size %d x %d to store q_lut information. \n", q_lut_texture_size, q_lut_texture_size);
    data_on_gpu->q_lut_texture_size = q_lut_texture_size;

    temp_memory = (float*)malloc(sizeof(float)*q_lut_texture_size*q_lut_texture_size);
    if(!temp_memory){
		printf("Couldn't allocate memory for q_lut texture...Exiting\n");
		exit(-1);
    }
    memset(temp_memory, 0, sizeof(float)*q_lut_texture_size*q_lut_texture_size); 
    memcpy(temp_memory, bspd->q_lut, sizeof(float)*num_voxels_per_region*64);

    printf("Transferrring the q_lut to the GPU. \n");
    data_on_gpu->q_lut_stream = new ::brook::stream(::brook::getStreamType((float *)0), q_lut_texture_size, q_lut_texture_size, -1);
    streamRead(*(data_on_gpu->q_lut_stream), temp_memory);
    free((void *)temp_memory);

	/* Allocate memory for the coefficient values */
    int coeff_texture_size = (int)ceil(sqrt((double)bspd->num_knots*3));
    printf("Allocating 2D texture of size %d x %d to store coefficient information. \n", coeff_texture_size, coeff_texture_size);
	data_on_gpu->coeff_texture_size = coeff_texture_size;
	data_on_gpu->coeff_stream = new ::brook::stream(::brook::getStreamType((float *)0), coeff_texture_size, coeff_texture_size, -1);

	/* Allocate memory for the dx, dy, and dz streams */
	data_on_gpu->dx_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
	data_on_gpu->dy_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
	data_on_gpu->dz_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);

	/* Allocate memory for the diff, valid voxel, and mvr streams */
	data_on_gpu->diff_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
	data_on_gpu->valid_voxel_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
	data_on_gpu->mvr_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);

	/* Allocate memory on GPU to store the score and the number of valid voxels as single float4 elements. */
	data_on_gpu->score = new ::brook::stream(::brook::getStreamType((float4 *)0), 1, 1, -1);
	data_on_gpu->num_valid_voxels = new ::brook::stream(::brook::getStreamType((float4 *)0), 1, 1, -1);

	/* Allocate memory on the CPU to store the diff values read back from the GPU */
	data_on_gpu->diff = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
	if(!data_on_gpu->diff){
	    printf("Couldn't allocate memory on CPU for diff result. Exiting. \n");
	    exit(-1);
	}
	memset(data_on_gpu->diff, 0, sizeof(float)*volume_texture_size*volume_texture_size*4);

	/* Allocate memory on CPU to store the dxyz stream */
	for(i = 0; i < 3; i++){
		data_on_gpu->dxyz[i] = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
		if(!data_on_gpu->dxyz[i]){
			printf("Couldn't allocate memory on CPU for dxyz result. Exiting. \n");
			exit(-1);
		}
		memset(data_on_gpu->dxyz[i], 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
	}

	/* Allocate memory on the CPU to store the valid voxel values read back from the GPU */
	data_on_gpu->valid_voxels = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
	if(!data_on_gpu->valid_voxels){
	    printf("Couldn't allocate memory on CPU for valid voxel result. Exiting. \n");
	    exit(-1);
	}
	memset(data_on_gpu->valid_voxels, 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
}

void 
bspline_initialize_structure_to_store_data_from_gpu(Volume* fixed, BSPLINE_Parms *parms)
{
    /* Allocate memory for GPU-specific data structure within the parms data structure */
    parms->data_from_gpu = (BSPLINE_DATA_FROM_GPU *)malloc(sizeof(BSPLINE_DATA_FROM_GPU));
    BSPLINE_DATA_FROM_GPU *data_from_gpu = (BSPLINE_DATA_FROM_GPU *)&parms->data_from_gpu;

	int volume_texture_size = (int)ceil(sqrt((double)fixed->npix/4)); // Size of the texture to allocate

	/* Allocate memory on the CPU to store the diff values read back from the GPU */
	data_from_gpu->diff = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
	if(!data_from_gpu->diff){
	    printf("Couldn't allocate memory on CPU for diff result. Exiting. \n");
	    exit(-1);
	}
	memset(data_from_gpu->diff, 0, sizeof(float)*volume_texture_size*volume_texture_size*4);

	/* Allocate memory on CPU to store the dxyz stream */
	for(int i = 0; i < 3; i++){
		data_from_gpu->dxyz[i] = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
		if(!data_from_gpu->dxyz[i]){
			printf("Couldn't allocate memory on CPU for dxyz result. Exiting. \n");
			exit(-1);
		}
		memset(data_from_gpu->dxyz[i], 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
	}

	/* Allocate memory on the CPU to store the valid voxel values read back from the GPU */
	data_from_gpu->valid_voxels = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
	if(!data_from_gpu->valid_voxels){
	    printf("Couldn't allocate memory on CPU for valid voxel result. Exiting. \n");
	    exit(-1);
	}
	memset(data_from_gpu->valid_voxels, 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
}


/* Compute score on the GPU */
void 
bspline_score_on_gpu_reference (BSPLINE_Parms *parms, 
				Volume *fixed, Volume *moving, Volume *moving_grad)
    {
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    BSPLINE_DATA_ON_GPU* data_on_gpu = (BSPLINE_DATA_ON_GPU *)&parms->data_on_gpu; 
	BSPLINE_DATA_FROM_GPU* data_from_gpu = (BSPLINE_DATA_FROM_GPU *)&parms->data_from_gpu;

    int i;
    int ri, rj, rk;
    int fi, fj, fk;
    int p[3];
    int q[3];
    float dc_dv[3];
    int num_vox;
    int pidx, qidx;
    float ssd_grad_norm, ssd_grad_mean;

    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    
    iter++;

    /* Prepare the GPU to execute the dxyz kernel */
    float3 volume_dim; 
    volume_dim.x = (float)fixed->dim[0]; /* Read in the dimensions of the volume */
    volume_dim.y = (float)fixed->dim[1];
    volume_dim.z = (float)fixed->dim[2];

    float3 rdims;
    rdims.x = (float)bspd->rdims[0]; /* Read in the dimensions of the region */
    rdims.y = (float)bspd->rdims[1];
    rdims.z = (float)bspd->rdims[2];

    float3 vox_per_rgn;
    vox_per_rgn.x = (float)parms->vox_per_rgn[0]; /* Read in spacing between the control knots */
    vox_per_rgn.y = (float)parms->vox_per_rgn[1];
    vox_per_rgn.z = (float)parms->vox_per_rgn[2];

	float3 img_origin;
	img_origin.x = (float)parms->img_origin[0]; /* Read in the coordinates of the image origin */
	img_origin.y = (float)parms->img_origin[1];
	img_origin.z = (float)parms->img_origin[2];

	float3 img_offset;
	img_offset.x = (float)moving->offset[0]; /* Read in image offset */
	img_offset.y = (float)moving->offset[1];
	img_offset.z = (float)moving->offset[2];

	float3 pix_spacing;
	pix_spacing.x = (float)moving->pix_spacing[0]; /* Read in the voxel dimensions */
	pix_spacing.y = (float)moving->pix_spacing[1];
	pix_spacing.z = (float)moving->pix_spacing[2];

	float4 score, num_valid_voxels;

	LARGE_INTEGER clock_count, clock_frequency;
    double clock_start, clock_end;

	QueryPerformanceFrequency(&clock_frequency); // Get CPU frequency
    QueryPerformanceCounter(&clock_count); // Get CPU cycle counter
    clock_start = (double)clock_count.QuadPart;

	/* Transfer the new coefficient values provided by the optimizer to the GPU */
	streamRead(*(data_on_gpu->coeff_stream), bspd->coeff);

    /* Invoke kernel on the GPU to compute dxyz, that determines the influence of the control knots on voxels in the X, Y, and Z directions */
    compute_dxyz_kernel(*(data_on_gpu->c_lut_stream), 
		 *(data_on_gpu->q_lut_stream), 
		 *(data_on_gpu->coeff_stream), 
		 volume_dim, 
		 vox_per_rgn, 
		 rdims,
		 (float)data_on_gpu->volume_texture_size,
		 (float)data_on_gpu->c_lut_texture_size,
		 (float)data_on_gpu->q_lut_texture_size,
		 (float)data_on_gpu->coeff_texture_size,
		 0.0, // Compute influence in the X direction
		 64.0,
		 *(data_on_gpu->dx_stream));

    compute_dxyz_kernel(*(data_on_gpu->c_lut_stream), 
		 *(data_on_gpu->q_lut_stream), 
		 *(data_on_gpu->coeff_stream), 
		 volume_dim, 
		 vox_per_rgn, 
		 rdims,
		 (float)data_on_gpu->volume_texture_size,
		 (float)data_on_gpu->c_lut_texture_size,
		 (float)data_on_gpu->q_lut_texture_size,
		 (float)data_on_gpu->coeff_texture_size,
		 1.0, // Compute influence in the Y direction
		 64.0,
		 *(data_on_gpu->dy_stream));

    compute_dxyz_kernel(*(data_on_gpu->c_lut_stream), 
		 *(data_on_gpu->q_lut_stream), 
		 *(data_on_gpu->coeff_stream), 
		 volume_dim, 
		 vox_per_rgn, 
		 rdims,
		 (float)data_on_gpu->volume_texture_size,
		 (float)data_on_gpu->c_lut_texture_size,
		 (float)data_on_gpu->q_lut_texture_size,
		 (float)data_on_gpu->coeff_texture_size,
		 2.0,	// Compute influence in the Z direction
		 64.0,
		 *(data_on_gpu->dz_stream));

	/* Determine the voxels that will contribute to the score computation */
	compute_valid_voxels_kernel( *(data_on_gpu->dx_stream),
				*(data_on_gpu->dy_stream),
				*(data_on_gpu->dz_stream),
				volume_dim, // X, Y, Z dimensions of the volume
				img_origin, // X, Y, Z coordinates for the image origin 
				pix_spacing, // Dimensions of a single voxel
				img_offset, // Offset corresponding to the region of interest
				(float)data_on_gpu->volume_texture_size,
				*(data_on_gpu->valid_voxel_stream));

	/* Compute the intensity difference between the static and moving images */
	compute_diff_kernel(*(data_on_gpu->fixed_image_stream), 
				*(data_on_gpu->moving_image_stream), 
				*(data_on_gpu->dx_stream),
				*(data_on_gpu->dy_stream),
				*(data_on_gpu->dz_stream),
				volume_dim, // X, Y, Z dimensions of the volume
				img_origin, // X, Y, Z coordinates for the image origin 
				pix_spacing, // Dimensions of a single voxel
				img_offset, // Offset corresponding to the region of interest
				(float)data_on_gpu->volume_texture_size,
				8.0,
				4.0,
				*(data_on_gpu->diff_stream));

	/* Compute the mvr values for voxels */
	compute_mvr_kernel(*(data_on_gpu->dx_stream),
				  *(data_on_gpu->dy_stream),
				  *(data_on_gpu->dz_stream), 
				  volume_dim, // X, Y, Z dimensions of the volume
				  img_origin, // X, Y, Z coordinates for the image origin 
				  pix_spacing, // Dimensions of a single voxel
				  img_offset, // Offset corresponding to the region of interest
				  (float)data_on_gpu->volume_texture_size,  
				  *(data_on_gpu->mvr_stream));

	/* Update the gradients in th X, Y, and Z directions */
	compute_dc_dv_kernel(*(data_on_gpu->moving_grad_stream_x),
						*(data_on_gpu->diff_stream), 
						*(data_on_gpu->mvr_stream),
						volume_dim, // X, Y, Z dimensions of the volume
						img_origin, // X, Y, Z coordinates for the image origin 
						pix_spacing, // Dimensions of a single voxel
						img_offset, // Offset corresponding to the region of interest
						(float)data_on_gpu->volume_texture_size,  
						*(data_on_gpu->dx_stream));	

	compute_dc_dv_kernel(*(data_on_gpu->moving_grad_stream_y), 
						*(data_on_gpu->diff_stream),
						*(data_on_gpu->mvr_stream),
						volume_dim, // X, Y, Z dimensions of the volume
						img_origin, // X, Y, Z coordinates for the image origin 
						pix_spacing, // Dimensions of a single voxel
						img_offset, // Offset corresponding to the region of interest
						(float)data_on_gpu->volume_texture_size,  
						*(data_on_gpu->dy_stream));	

	compute_dc_dv_kernel(*(data_on_gpu->moving_grad_stream_z),
						*(data_on_gpu->diff_stream),
						*(data_on_gpu->mvr_stream),
						volume_dim, // X, Y, Z dimensions of the volume
						img_origin, // X, Y, Z coordinates for the image origin 
						pix_spacing, // Dimensions of a single voxel
						img_offset, // Offset corresponding to the region of interest
						(float)data_on_gpu->volume_texture_size,  
						*(data_on_gpu->dz_stream));	

	/* Compute diff*diff on the GPU and reduce it to a single value to compute the score */
	compute_diff_squared_kernel(*(data_on_gpu->diff_stream), *(data_on_gpu->diff_stream));
	compute_score_kernel(*(data_on_gpu->diff_stream), *(data_on_gpu->score));
	
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
	printf("Time needed to compute the kernels on the GPU = %f \n", double(clock_end - clock_start)/(double)clock_frequency.QuadPart);

	/* Read the diff_squared and valid voxel streams back from the GPU. */
	// streamWrite(*(data_on_gpu->diff_stream), data_on_gpu->diff);
	streamWrite(*(data_on_gpu->valid_voxel_stream), data_on_gpu->valid_voxels);
	compute_num_valid_voxels_kernel(*(data_on_gpu->valid_voxel_stream), *(data_on_gpu->num_valid_voxels));

	streamWrite(*(data_on_gpu->score), &score);
	streamWrite(*(data_on_gpu->num_valid_voxels), &num_valid_voxels);
	
	/* Read back the dc_dv values from the GPU */
	streamWrite(*(data_on_gpu->dx_stream), data_on_gpu->dxyz[0]);
    streamWrite(*(data_on_gpu->dy_stream), data_on_gpu->dxyz[1]);
    streamWrite(*(data_on_gpu->dz_stream), data_on_gpu->dxyz[2]);
	
	int vox = -1; 
    ssd->score = 0;
    memset (ssd->grad, 0, bspd->num_coeff * sizeof(float));
    num_vox = 0;
    for (rk = 0, fk = parms->roi_offset[2]; rk < parms->roi_dim[2]; rk++, fk++) {
		p[2] = rk / parms->vox_per_rgn[2];
		q[2] = rk % parms->vox_per_rgn[2];
		for (rj = 0, fj = parms->roi_offset[1]; rj < parms->roi_dim[1]; rj++, fj++) {
			p[1] = rj / parms->vox_per_rgn[1];
			q[1] = rj % parms->vox_per_rgn[1];
			for (ri = 0, fi = parms->roi_offset[0]; ri < parms->roi_dim[0]; ri++, fi++) {
				vox++;
				p[0] = ri / parms->vox_per_rgn[0];
				q[0] = ri % parms->vox_per_rgn[0];

				pidx = ((p[2] * bspd->rdims[1] + p[1]) * bspd->rdims[0]) + p[0];
				qidx = ((q[2] * parms->vox_per_rgn[1] + q[1]) * parms->vox_per_rgn[0]) + q[0];

				if(data_on_gpu->valid_voxels[vox] == 0.0) continue;
				else{
					dc_dv[0] = data_on_gpu->dxyz[0][vox];
					dc_dv[1] = data_on_gpu->dxyz[1][vox];
					dc_dv[2] = data_on_gpu->dxyz[2][vox];
					
					bspline_update_grad_b (parms, pidx, qidx, dc_dv);
					num_vox ++;
				}
			}
		}
    }

    //dump_coeff (bspd, "coeff.txt");

    /* Normalize score for MSE */
    // ssd->score = ssd->score / num_vox;
	ssd->score = (score.x + score.y + score.z + score.w)/(num_valid_voxels.x + num_valid_voxels.y + num_valid_voxels.z + num_valid_voxels.w);
    for (i = 0; i < bspd->num_coeff; i++) {
	// ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
		ssd->grad[i] = 2 * ssd->grad[i] / (num_valid_voxels.x + num_valid_voxels.y + num_valid_voxels.z + num_valid_voxels.w);
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bspd->num_coeff; i++) {
		ssd_grad_mean += ssd->grad[i];
		ssd_grad_norm += fabs (ssd->grad[i]);
    }

#if defined (commentout)
    printf ("Single iteration CPU [b] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
    printf ("NUM_VOX = %d\n", num_vox);
    printf ("MSE = %g\n", ssd->score);
    printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    printf ("GRAD_NORM = %g\n", ssd_grad_norm);
#endif

	// printf("Time on CPU to execute the interpolate function = %f\n", cpu_cycles/(double)clock_frequency.QuadPart);
    printf ("GET VALUE+DERIVATIVE: %6.3f [%6d] %6.3f %6.3f \n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm);

}

} /* extern "C" */

