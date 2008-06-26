/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    REFS:
    http://en.wikipedia.org/wiki/B-spline
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
    http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    ----------------------------------------------------------------------- */
#include <brook/brook.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "volume.h"
#include "readmha.h"
#include "bspline_opts.h"
#include "bspline.h"

typedef struct t_s {
	::brook::stream *out_stream;
	::brook::stream *in_stream;
    } TEST_STRUCT;

void  compute_dxyz_b (::brook::stream q_lut,
		::brook::stream c_lut,
		::brook::stream coeff_lut,
		const float3  dim,
		const float3  rdims,
		const float3  int_spacing,
		const float  volume_texture_size,
		const float  q_lut_texture_size,
		const float  c_lut_texture_size,
		const float  coeff_lut_texture_size,
		const float  xyz,
		const float  sixty_four,
		::brook::stream result);
void  compute_dxyz_reference (::brook::stream lut,
		::brook::stream coeff,
		const float3  dim,
		const float3  cdims,
		const float3  int_spacing,
		const float  volume_texture_size,
		const float  lut_texture_size,
		const float  coeff_texture_size,
		const float  xyz,
		const float  four,
		::brook::stream result);

void toy_a(float loop_index, ::brook::stream result);
void toy_b(::brook::stream in_stream, float loop_index, ::brook::stream result);
void my_sum(::brook::stream foo, ::brook::stream bar);
void init(::brook::stream result);

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

inline void
bspline_update_grad (BSPLINE_Score* ssd, BSPLINE_Data* bspd, 
		     int p[3], int qidx, float dc_dv[3])
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bspd->q_lut[qidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = (p[2] + k) * bspd->cdims[1] * bspd->cdims[0]
			+ (p[1] + j) * bspd->cdims[0]
			+ (p[0] + i);
		cidx = cidx * 3;
		ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
		ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
		ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
		m ++;
	    }
	}
    }
}

extern "C" {

void 
bspline_score_on_gpu_reference (BSPLINE_Score *ssd, 
				Volume *fixed, Volume *moving, Volume *moving_grad, 
				BSPLINE_Data *bspd, BSPLINE_Parms *parms)
{
    int i, j, k;
    int v, mv;
    int mx, my, mz;
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    int num_vox;
    int qidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;


    ssd->score = 0;
    num_vox = 0;
    memset (ssd->grad, 0, bspd->num_knots * sizeof(float));

    /* Read in volume dimensions */
    float3 dim; 
    dim.x = (float)fixed->dim[0];
    dim.y = (float)fixed->dim[1];
    dim.z = (float)fixed->dim[2];

    /* Read in knot dimensions */
    float3 cdims;
    cdims.x = bspd->cdims[0];
    cdims.y = bspd->cdims[1];
    cdims.z = bspd->cdims[2];

    /* Read in spacing between the control knots */
    float3 int_spacing;
    int_spacing.x = parms->int_spacing[0];
    int_spacing.y = parms->int_spacing[1];
    int_spacing.z = parms->int_spacing[2];

    /* Compute the size of the texture to hold the x, y, z coefficients of each knot and allocate memory */
    int coeff_texture_size = (int)ceil(sqrt((double)bspd->num_knots));
    printf("Allocating 2D texture of size %d to store coefficient information \n", coeff_texture_size);
    float* temp_coeff = (float*)malloc(sizeof(float)*coeff_texture_size*coeff_texture_size);
    if(!temp_coeff){
	printf("Couldn't allocate memory for coefficient texture...Exiting\n");
	exit(-1);
    }
    memset(temp_coeff, 0, sizeof(float)*coeff_texture_size*coeff_texture_size);
    memcpy(temp_coeff, bspd->coeff, sizeof(float)*bspd->num_knots); // Copy bspd->coeff values
    
    /* Compute the size of the texture to hold the lut */
    int lut_texture_size = 
	(int)ceil(sqrt((double)(parms->int_spacing[0]*parms->int_spacing[1]*parms->int_spacing[2]*64)));
    printf("Allocating 2D texture of size %d to store lut information \n", lut_texture_size);
    float* temp_lut = (float*)malloc(sizeof(float)*lut_texture_size*lut_texture_size);
    if(!temp_lut){
	printf("Couldn't allocate memory for lut texture...Exiting\n");
	exit(-1);
    }
    memset(temp_lut, 0, sizeof(float)*lut_texture_size*lut_texture_size); // Initialize memory 
    memcpy(temp_lut, 
	bspd->q_lut, 
	sizeof(float)*parms->int_spacing[0]*parms->int_spacing[1]*parms->int_spacing[2]*64); // Copy lut

    /* Compute the size of the texture to hold the moving and static images */
    int volume_texture_size = (int)ceil(sqrt((double)fixed->npix/4)); // Size of the texture to allocate
    printf("Allocating 2D texture of size %d to store volume information \n", volume_texture_size);
    float* temp_fixed_image = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
    float* temp_moving_image = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
    if(!temp_fixed_image || !temp_moving_image){
	printf("Couldn't allocate texture memory for volume...Exiting\n");
	exit(-1);
    }
    memset(temp_fixed_image, 0, sizeof(float)*volume_texture_size*volume_texture_size*4); // Initialize memory 
    memcpy(temp_fixed_image, fixed->img, sizeof(float)*fixed->npix); // Copy fixed image
    memset(temp_moving_image, 0, sizeof(float)*volume_texture_size*volume_texture_size*4); // Initialize memory 
    memcpy(temp_moving_image, moving->img, sizeof(float)*moving->npix); // Copy moving image

    /* Allocate memory to store the result from the GPU */
    float* my_dxyz[3]; // Data structure to store the x, y, z displacment vectors
    for(i = 0; i < 3; i++){
	my_dxyz[i] = (float*)malloc(sizeof(float)*volume_texture_size*volume_texture_size*4);
	if(!my_dxyz){
	    printf("Couldn't allocate texture memory for dxyz result...Exiting\n");
	    exit(-1);
	}
	memset(my_dxyz[i], 0, sizeof(float)*volume_texture_size*volume_texture_size*4); // Initialize memory
    }

    /* Allocate memory for GPU streams */
    printf("Allocating memory for GPU streams. \n");
    ::brook::stream coeff_stream(::brook::getStreamType((float *)0), coeff_texture_size, coeff_texture_size, -1);
    ::brook::stream lut_stream(::brook::getStreamType((float *)0), lut_texture_size, lut_texture_size, -1);
    ::brook::stream dx_stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
    ::brook::stream dy_stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);
    ::brook::stream dz_stream(::brook::getStreamType((float4 *)0), volume_texture_size, volume_texture_size, -1);

    /* Copy streams to the GPU */
    printf("copying streams to the GPU. \n");
    streamRead(coeff_stream, temp_coeff); // Copy coefficient stream
    streamRead(lut_stream, temp_lut); // Copy lut stream

    printf("Executing kernels on the GPU. \n");
    start_clock = clock();

#if defined (commentout)
    /* Invoke kernel on the GPU to compute dxyz, xyz = 0 for x, 1 for y and 2 for z */
    compute_dxyz_reference(lut_stream, coeff_stream, dim, cdims, int_spacing, volume_texture_size, lut_texture_size, 
	coeff_texture_size, 0, dx_stream);

    compute_dxyz_reference(lut_stream, coeff_stream, dim, cdims, int_spacing, volume_texture_size, lut_texture_size, 
	coeff_texture_size, 1, dy_stream);

    compute_dxyz_reference(lut_stream, coeff_stream, dim, cdims, int_spacing, volume_texture_size, lut_texture_size, 
	coeff_texture_size, 2, dz_stream);
#endif

    end_clock = clock();
    printf("Interpolation time on the GPU = %f\n", (double)(end_clock - start_clock)/CLOCKS_PER_SEC);

    /* Read dx, dy, and dz streams back from the GPU */
    streamWrite(dx_stream, my_dxyz[0]);
    streamWrite(dy_stream, my_dxyz[1]);
    streamWrite(dz_stream, my_dxyz[2]);

    /* Copy streams to dxyz structure */
    float *dxyz = (float *) malloc (3*sizeof(float)*fixed->npix);
    for(i = 0; i < fixed->npix; i++){
	dxyz[3*i] = my_dxyz[0][i];
	dxyz[3*i+1] = my_dxyz[1][i];
	dxyz[3*i+2] = my_dxyz[2][i];
    }

    /* Compute score and gradient based on correspondences */
    for (v = 0, k = 0; k < fixed->dim[2]; k++) {
	p[2] = k / parms->int_spacing[2];
	q[2] = k % parms->int_spacing[2];
	for (j = 0; j < fixed->dim[1]; j++) {
	    p[1] = j / parms->int_spacing[1];
	    q[1] = j % parms->int_spacing[1];
	    for (i = 0; i < fixed->dim[0]; i++, v++) {
		p[0] = i / parms->int_spacing[0];
		q[0] = i % parms->int_spacing[0];
		qidx = q[2] * parms->int_spacing[1] * parms->int_spacing[0]
			+ q[1] * parms->int_spacing[0] + q[0];

		/* Nearest neighbor interpolation & boundary checking */
		/* Nearest neighbor interpolation & boundary checking */
		mz = k + round_int (dxyz[3*v+2]);
		if (mz < 0 || mz >= moving->dim[2]) continue;
		my = j + round_int (dxyz[3*v+1]);
		if (my < 0 || my >= moving->dim[1]) continue;
		mx = i + round_int (dxyz[3*v]);
		if (mx < 0 || mx >= moving->dim[0]) continue;

		/* v is linear index of this voxel in fixed image */
		/* mv is linear index of matching voxel in moving image */
		mv = (mz * moving->dim[1] + my) * moving->dim[0] + mx;
		diff = f_img[v] - m_img[mv];

		/* dc_dv is partial derivative of cost function with respect to 
		    deformation vector. */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */

		/* Combine dc_dv with dv_dp, where dv_dp is partial derivative 
		    of deformation vector with respect to bspline coefficient.
		    The inner product of dc_dv and dv_dp is the gradient. */
		bspline_update_grad (ssd, bspd, p, qidx, dc_dv);
		
		/* Finally, accumulate the score as mean squared difference */
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bspd->num_knots; i++) {
	ssd->grad[i] /= num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bspd->num_knots; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }
    printf ("MSE = %g\n", ssd->score);
    printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    printf ("GRAD_NORM = %g\n", ssd_grad_norm);

    free (dxyz);
    free(temp_coeff);
    free(temp_lut);
}

void run_toy_kernel_a(void){
    float out[40000];
    int i;
    float4 this_sum;

    for(i = 0; i < 40000; i++)
	out[i] = 0.0;

    printf("Allocating stream memory \n");
    ::brook::stream out_stream(::brook::getStreamType((float4 *)0), 100, 100, -1);
    ::brook::stream this_element(::brook::getStreamType((float4 *)0), 1, 1, -1);

    init(out_stream);
    for(i = 0; i < 1000; i++){
	printf("Kernel iteration %d \n", i);
	toy_b(out_stream, 20, out_stream);
	// run_toy_kernel_c(test_struct);
    }
    /* printf("Executing kernel \n");
    toy_a(20, out_stream);
    my_sum(out_stream, this_element); */

    streamWrite(out_stream, out);
    // streamWrite(this_element, &this_sum);

    printf("\n");
    for(i = 0; i < 40000; i++)
	printf("%f ", out[i]);
    printf("\n");

    // printf("Sum = %f \n", (this_sum.x + this_sum.y + this_sum.z + this_sum.w));
    getchar();

}

void run_toy_kernel_b(float i, ::brook::stream in_stream, ::brook::stream out_stream, ::brook::stream this_element){
    // toy_a(i, out_stream);
    toy_b(out_stream, 20, out_stream);
}

void run_toy_kernel_c(TEST_STRUCT *this_test_struct){
    toy_b(*(this_test_struct->out_stream), 20, *(this_test_struct->out_stream)); 
}

void run_toy_kernel(void){
    float out[40000];
    float in[40000];
    int i;
    float4 this_sum;
    
    TEST_STRUCT *test_struct = (TEST_STRUCT *)malloc(sizeof(TEST_STRUCT));

    for(i = 0; i < 40000; i++){
	out[i] = 0.0;
	in[i] = 10.0;
    }

    printf("Allocating stream memory \n");
    /* ::brook::stream out_stream(::brook::getStreamType((float4 *)0), 10, 10, -1);
    ::brook::stream this_element(::brook::getStreamType((float4 *)0), 1, 1, -1);
    ::brook::stream in_stream(::brook::getStreamType((float4 *)0), 10, 10, -1); */


    /* ::brook::stream *out_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), 10, 10, -1);
    ::brook::stream this_element(::brook::getStreamType((float4 *)0), 1, 1, -1);
    ::brook::stream *in_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), 10, 10, -1); */
    test_struct->out_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), 100, 100, -1);
    test_struct->in_stream = new ::brook::stream(::brook::getStreamType((float4 *)0), 100, 100, -1);
    ::brook::stream this_element(::brook::getStreamType((float4 *)0), 1, 1, -1);

    printf("Transferring stream to GPU \n");
    // streamRead(test_stream, in);
    streamRead(*(test_struct->in_stream), in);
    printf("Done \n");
    init(*(test_struct->out_stream));
    for(i = 0; i < 1000; i++){
	printf("Kernel iteration %d \n", i);
	printf("Kernel iteration %d \n", i);
	run_toy_kernel_b(1, *(test_struct->in_stream), *(test_struct->out_stream), this_element);
	// run_toy_kernel_c(test_struct);
    }

    streamWrite(*(test_struct->out_stream), out);
    // streamWrite(this_element, &this_sum);

    for(i = 0; i < 40000; i++)
	printf("%f ", out[i]);
    printf("\n");

    getchar();
}

} /* extern "C" */

