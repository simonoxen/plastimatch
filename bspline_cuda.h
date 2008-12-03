/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_cuda_h_
#define _bspline_cuda_h_

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "bspline.h"

#if defined __cplusplus
extern "C" {
#endif

	typedef struct BSPLINE_CUDA_Data_struct BSPLINE_CUDA_Data;
	struct BSPLINE_CUDA_Data_struct {
		size_t image_size;
		float *fixed_image;  // The fixed image
		float *moving_image; // The moving image
		float *moving_grad_x; // Streams to store the gradient of the moving image in the X, Y, and Z directions 
		float *moving_grad_y; 
		float *moving_grad_z; 
		int   *c_lut; // The c_lut indicating which control knots affect voxels within a region
		float *q_lut; // The q_lut indicating the distance of a voxel to each of the 64 control knots
		float *coeff; // The coefficient stream indicating the x, y, z coefficients of each control knot
		float *dx; // Streams to store voxel displacement/gradient values in the X, Y, and Z directions 
		float *dy; 
		float *dz; 
		float *mvr; // Streams to store the mvr values 
		float *diff; // Stream to store the correspondence values in the moving image---for debug purposes only
		float *valid_voxel; // Stream to indicate if a voxel should take part in the score computation or not
		float *partial_sum; // Stream for storing the partial sums during reductions 
		float *sum_element;
	};

	void bspline_cuda_score_mse(
		BSPLINE_Parms *parms, 
		BSPLINE_Xform* bxf, 
		Volume *fixed, 
		Volume *moving, 
		Volume *moving_grad);

	// Simple utility function to check for CUDA runtime errors.
	void checkCUDAError(const char *msg);  

	// Allocate memory on the GPU and copy all necessary data to the GPU.
	void bspline_cuda_initialize(
		Volume *fixed,
		Volume *moving,
		Volume *moving_grad,
		BSPLINE_Xform *bxf,
		BSPLINE_Parms *parms);

	void bspline_cuda_run_kernels(
		Volume *fixed,
		Volume *moving,
		Volume *moving_grad,
		BSPLINE_Xform *bxf,
		BSPLINE_Parms *parms,
		float *host_diff,
		float *host_dc_dv_x,
		float *host_dc_dv_y,
		float *host_dc_dv_z);

	void bspline_cuda_clean_up();

#if defined __cplusplus
}
#endif

#endif
