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
check_values(float cpu_value, float gpu_value){
    printf("CPU: %f, GPU: %f; diff = %f \n", cpu_value, gpu_value, abs(cpu_value - gpu_value));
    }

void 
bspline_initialize_streams_on_gpu(Volume *fixed, Volume *moving, BSPLINE_Parms *parms){
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

	/* Allocate memory on the CPU to store the dxyz values read back from the GPU */
	for(int i = 0; i < 3; i++){
	data_on_gpu->my_dxyz[i] = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
	if(!data_on_gpu->my_dxyz){
	    printf("Couldn't allocate texture memory for dxyz result. Exiting. \n");
	    exit(-1);
	}
	memset(data_on_gpu->my_dxyz[i], 0, sizeof(float)*volume_texture_size*volume_texture_size*4);
    }
}

void 
bspline_score_on_gpu_reference (BSPLINE_Parms *parms, 
				Volume *fixed, Volume *moving, Volume *moving_grad)
    {
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    BSPLINE_DATA_ON_GPU* data_on_gpu = (BSPLINE_DATA_ON_GPU *)&parms->data_on_gpu; 

    int i;
    int ri, rj, rk;
    int fi, fj, fk, fv;
    float mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int mif, mjf, mkf, mvf;  /* Floor */
    int mir, mjr, mkr, mvr;  /* Round */
    int p[3];
    int q[3];
    float dxyz[3];
    float diff;
    float dc_dv[3];
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    int num_vox;
    int pidx, qidx;
    float ssd_grad_norm, ssd_grad_mean;
    
	/* Some variables for timing */
	clock_t start_clock, end_clock;
	
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

    start_clock = clock();
    iter++;

    /* Prepare the GPU to execute the dxyz kernel */

    /* Read in the dimensions of the fixed and moving volumes */
    float3 volume_dim; 
    volume_dim.x = (float)fixed->dim[0];
    volume_dim.y = (float)fixed->dim[1];
    volume_dim.z = (float)fixed->dim[2];

    /* Read in the dimensions of the region */
    float3 rdims;
    rdims.x = (float)bspd->rdims[0];
    rdims.y = (float)bspd->rdims[1];
    rdims.z = (float)bspd->rdims[2];

    // Read in spacing between the control knots
    float3 vox_per_rgn;
    vox_per_rgn.x = (float)parms->vox_per_rgn[0];
    vox_per_rgn.y = (float)parms->vox_per_rgn[1];
    vox_per_rgn.z = (float)parms->vox_per_rgn[2];


	/* Transfer the new coefficient values to the GPU */
	streamRead(*(data_on_gpu->coeff_stream), bspd->coeff);

    /* Invoke kernel on the GPU to compute dxyz, xyz = 0 for x, 1 for y and 2 for z */
	// printf("Calling kernels. \n");
	LARGE_INTEGER clock_count;
    LARGE_INTEGER clock_frequency;
    double clock_start, clock_end;
    double kernel_cycles = 0.0;
	double cpu_cycles = 0.0;

    QueryPerformanceFrequency(&clock_frequency); // Get CPU frequency
    QueryPerformanceCounter(&clock_count); // Get CPU cycle counter
    clock_start = (double)clock_count.QuadPart;

    compute_dxyz(*(data_on_gpu->c_lut_stream), 
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

    compute_dxyz(*(data_on_gpu->c_lut_stream), 
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

    compute_dxyz(*(data_on_gpu->c_lut_stream), 
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
	// printf("Done calling kernels. \n");

    /* Read dx, dy, and dz streams back from the GPU. */
    streamWrite(*(data_on_gpu->dx_stream), data_on_gpu->my_dxyz[0]);
    streamWrite(*(data_on_gpu->dy_stream), data_on_gpu->my_dxyz[1]);
    streamWrite(*(data_on_gpu->dz_stream), data_on_gpu->my_dxyz[2]);
	// printf("Done read back from GPU. \n");
	QueryPerformanceCounter(&clock_count);
    clock_end = (double)clock_count.QuadPart;
    kernel_cycles +=  (clock_end - clock_start);
	printf("Time on GPU to execute the interpolate function = %f\n", kernel_cycles/(double)clock_frequency.QuadPart);

    ssd->score = 0;
    memset (ssd->grad, 0, bspd->num_coeff * sizeof(float));
    num_vox = 0;
    for (rk = 0, fk = parms->roi_offset[2]; rk < parms->roi_dim[2]; rk++, fk++) {
	p[2] = rk / parms->vox_per_rgn[2];
	q[2] = rk % parms->vox_per_rgn[2];
	fz = parms->img_origin[2] + parms->img_spacing[2] * fk;
	for (rj = 0, fj = parms->roi_offset[1]; rj < parms->roi_dim[1]; rj++, fj++) {
	    p[1] = rj / parms->vox_per_rgn[1];
	    q[1] = rj % parms->vox_per_rgn[1];
	    fy = parms->img_origin[1] + parms->img_spacing[1] * fj;
	    for (ri = 0, fi = parms->roi_offset[0]; ri < parms->roi_dim[0]; ri++, fi++) {
		p[0] = ri / parms->vox_per_rgn[0];
		q[0] = ri % parms->vox_per_rgn[0];
		fx = parms->img_origin[0] + parms->img_spacing[0] * fi;

		/* Get B-spline deformation vector */
		pidx = ((p[2] * bspd->rdims[1] + p[1]) * bspd->rdims[0]) + p[0];
		qidx = ((q[2] * parms->vox_per_rgn[1] + q[1]) * parms->vox_per_rgn[0]) + q[0];

#if defined (commentout)
		/* Uncomment to compare the dxyz values generated by the CPU and GPU */
		QueryPerformanceCounter(&clock_count);
		clock_start = (double)clock_count.QuadPart;
		bspline_interp_pix_b (dxyz, bspd, pidx, qidx);
		
		if(iter == 15){
		    check_values(dxyz[0], my_dxyz[0][num_vox]);
			check_values(dxyz[1], my_dxyz[1][num_vox]);
			check_values(dxyz[2], my_dxyz[2][num_vox]);
		}
		
		QueryPerformanceCounter(&clock_count);
		clock_end = (double)clock_count.QuadPart;
		cpu_cycles +=  (clock_end - clock_start);
#endif
		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		/* Find correspondence in moving image */
		mx = fx + data_on_gpu->my_dxyz[0][num_vox];
		mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
		if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

		my = fy + data_on_gpu->my_dxyz[1][num_vox];
		mj = (my - moving->offset[1]) / moving->pix_spacing[1];
		if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

		mz = fz + data_on_gpu->my_dxyz[2][num_vox];
		mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
		if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

		/* Compute interpolation fractions */
		clamp_and_interpolate (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
		clamp_and_interpolate (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
		clamp_and_interpolate (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

		/* Compute moving image intensity using linear interpolation */
		mvf = (mkf * moving->dim[1] + mjf) * moving->dim[0] + mif;
		m_x1y1z1 = fx1 * fy1 * fz1 * m_img[mvf];
		m_x2y1z1 = fx2 * fy1 * fz1 * m_img[mvf+1];
		m_x1y2z1 = fx1 * fy2 * fz1 * m_img[mvf+moving->dim[0]];
		m_x2y2z1 = fx2 * fy2 * fz1 * m_img[mvf+moving->dim[0]+1];
		m_x1y1z2 = fx1 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]];
		m_x2y1z2 = fx2 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1];
		m_x1y2z2 = fx1 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]];
		m_x2y2z2 = fx2 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1];
		m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1 
			+ m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;

		/* Compute intensity difference */
		diff = f_img[fv] - m_val;

		/* Compute spatial gradient using nearest neighbors */
		mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;
		dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
		bspline_update_grad_b (parms, pidx, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
		}
    }

    //dump_coeff (bspd, "coeff.txt");

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bspd->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bspd->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }

    end_clock = clock();
#if defined (commentout)
    printf ("Single iteration CPU [b] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
    printf ("NUM_VOX = %d\n", num_vox);
    printf ("MSE = %g\n", ssd->score);
    printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    printf ("GRAD_NORM = %g\n", ssd_grad_norm);
#endif

	printf("Time on CPU to execute the interpolate function = %f\n", cpu_cycles/(double)clock_frequency.QuadPart);
    printf ("GET VALUE+DERIVATIVE: %6.3f [%6d] %6.3f %6.3f \n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm);
}

} /* extern "C" */

