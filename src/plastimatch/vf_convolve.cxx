/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "math_util.h"
#include "plm_int.h"
#include "volume.h"
#include "vf_convolve.h"

void
vf_convolve_x (Volume* vf_out, Volume* vf_in, float* ker, int width)
{
    int v,x,y,z;
    int half_width;
    float *in_img = (float*) vf_in->img;
    float *out_img = (float*) vf_out->img;

    half_width = width / 2;

    for (v = 0, z = 0; z < vf_in->dim[2]; z++) {
	for (y = 0; y < vf_in->dim[1]; y++) {
	    for (x = 0; x < vf_in->dim[0]; x++, v++) {
		int i, i1;	    /* i is the offset in the vf */
		int j, j1, j2;	    /* j is the index of the kernel */
		int d;		    /* d is the vector field direction */
		float *vin = &in_img[3*v];
		float *vout = &out_img[3*v];

		j1 = x - half_width;
		j2 = x + half_width;
		if (j1 < 0) j1 = 0;
		if (j2 >= vf_in->dim[0]) {
		    j2 = vf_in->dim[0] - 1;
		}
		i1 = j1 - x;
		j1 = j1 - x + half_width;
		j2 = j2 - x + half_width;

		for (d = 0; d < 3; d++) {
		    vout[d] = (float) 0.0;
		    for (i = i1, j = j1; j <= j2; i++, j++) {
			vout[d] += ker[j] * (vin+i*3)[d];
		    }
		}
	    }
	}
    }
}

void
vf_convolve_y (Volume* vf_out, Volume* vf_in, float* ker, int width)
{
    int v,x,y,z;
    int half_width;
    float *in_img = (float*) vf_in->img;
    float *out_img = (float*) vf_out->img;

    half_width = width / 2;

    for (v = 0, z = 0; z < vf_in->dim[2]; z++) {
	for (y = 0; y < vf_in->dim[1]; y++) {
	    for (x = 0; x < vf_in->dim[0]; x++, v++) {
		int i, i1;	    /* i is the offset in the vf */
		int j, j1, j2;	    /* j is the index of the kernel */
		int d;		    /* d is the vector field direction */
		float *vin = &in_img[3*v];
		float *vout = &out_img[3*v];

		j1 = y - half_width;
		j2 = y + half_width;
		if (j1 < 0) j1 = 0;
		if (j2 >= vf_in->dim[1]) {
		    j2 = vf_in->dim[1] - 1;
		}
		i1 = j1 - y;
		j1 = j1 - y + half_width;
		j2 = j2 - y + half_width;

		for (d = 0; d < 3; d++) {
		    vout[d] = (float) 0.0;
		    for (i = i1, j = j1; j <= j2; i++, j++) {
			vout[d] += ker[j] * (vin+i*vf_in->dim[0]*3)[d];
		    }
		}
	    }
	}
    }
}

void
vf_convolve_z (Volume* vf_out, Volume* vf_in, float* ker, int width)
{
    int v,x,y,z;
    int half_width;
    float *in_img = (float*) vf_in->img;
    float *out_img = (float*) vf_out->img;

    half_width = width / 2;

    for (v = 0, z = 0; z < vf_in->dim[2]; z++) {
	for (y = 0; y < vf_in->dim[1]; y++) {
	    for (x = 0; x < vf_in->dim[0]; x++, v++) {
		int i, i1;	    /* i is the offset in the vf */
		int j, j1, j2;	    /* j is the index of the kernel */
		int d;		    /* d is the vector field direction */
		float *vin = &in_img[3*v];
		float *vout = &out_img[3*v];

		j1 = z - half_width;
		j2 = z + half_width;
		if (j1 < 0) j1 = 0;
		if (j2 >= vf_in->dim[2]) {
		    j2 = vf_in->dim[2] - 1;
		}
		i1 = j1 - z;
		j1 = j1 - z + half_width;
		j2 = j2 - z + half_width;

		for (d = 0; d < 3; d++) {
		    vout[d] = (float) 0.0;
		    for (i = i1, j = j1; j <= j2; i++, j++) {
			vout[d] += ker[j] * (vin+i*vf_in->dim[0]*vf_in->dim[1]*3)[d];
		    }
		}
	    }
	}
    }
}

