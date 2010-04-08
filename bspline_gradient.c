/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "bspline.h"
#include "bspline_gradient.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "timer.h"

/*internal use only - get k-th component (k=0,1,2) of 
the vector field at location r using pre-rendered vf */
static float f2( int k, int r[3], float  *vf, int *dims )
{
int d;
d = r[2] * dims[0] * dims[1] + r[1] * dims[0] + r[0];
return vf[3*d+k];
}

/*
first derivative of the vector field du_i/dx_i using pre-rendered vf
calculated at position r[3] = (ri rj rk) in voxels */
float bspline_gradient_1st_derivative(  int i,  int r[3], float h[3], float *vf, int *dims )
{
int r1[3], r2[3];
int d;

for(d=0; d<3; d++) {  r1[d]=r[d]; r2[d]=r[d]; }
	r1[i] ++;
	r2[i] --;

return (f2(i,r1, vf, dims)-f2(i,r2, vf, dims))/(2.*h[i]);
}

/* second derivative of k-th component of vector field wrt x_i and x_j, 
d2u_k/(dx_i dx_j) calculated at position r[3] = (ri rj rk) in voxels 
using pre-rendered vf */
float bspline_gradient_2nd_derivative( int k, int i, int j,  int r[3], float h[3], float *vf, int *dims )
{
int r1[3], r2[3], r3[3], r4[3], r5[3], r6[3];
int d;

if (i==j) {
	for(d=0; d<3; d++) {  r1[d]=r[d]; r2[d]=r[d]; }
	r1[i] ++;
	r2[i] --;
    return  ( f2(k, r1, vf, dims) + f2(k, r2, vf, dims) - 2*f2(k, r, vf, dims ) ) / (h[i]*h[i]);
} else {
	for(d=0; d<3; d++) { 
							r1[d]=r[d]; 
							r2[d]=r[d]; 
							r3[d]=r[d]; 
							r4[d]=r[d];
							r5[d]=r[d]; 
							r6[d]=r[d]; 
							}
	/* voxel not used*/	r1[j]++;	r2[j]++; r2[i]++;
	r3[i]--;			/* r[] */		r4[i]++;
	r5[i]--; r5[j]--;   r6[j]--;  /*voxel not used */

	return ( -f2(k,r1,vf, dims)+f2(k,r2,vf,dims)-f2(k,r3,vf,dims)
			 +2*f2(k,r,vf,dims)
			 -f2(k,r4,vf, dims)+f2(k,r5,vf,dims)-f2(k,r6,vf,dims))/(2.*h[i]*h[j]);
}
}


void
bspline_gradient_score (
    BSPLINE_Parms *parms, 
    Bspline_state *bst, 
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    float grad_score;
	int ri, rj, rk;
	int fi, fj, fk;
	//int mi, mj, mk;
    //float fx, fy, fz;
    //float mx, my, mz;
    int p[3];
    int q[3];
	int rvec[3]; // ri rj rk as a vector
    float dxyz[3];
    int qidx;
	//float diff[3];
	float dc_dv[3];
	float dux_dx[3];
	int num_vox, nv;
	int d,d1,d2,k;
	Timer timer;
	double interval;
    float *vf; float du;

    grad_score = 0;
	num_vox = 0;
    nv = 0;

	plm_timer_start (&timer);

    vf = (float*) malloc( 3*bxf->roi_dim[0]*bxf->roi_dim[1]*bxf->roi_dim[2] *sizeof(float)); 

	printf("---- YOUNG MODULUS %f\n", parms->young_modulus);

	/* rendering the entire field */
	for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
		p[2] = rk / bxf->vox_per_rgn[2];
		q[2] = rk % bxf->vox_per_rgn[2];
		for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
			p[1] = rj / bxf->vox_per_rgn[1];
			q[1] = rj % bxf->vox_per_rgn[1];
	        for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
			p[0] = ri / bxf->vox_per_rgn[0];
			q[0] = ri % bxf->vox_per_rgn[0];
			
			/* Get B-spline deformation vector */
			qidx = INDEX_OF (q, bxf->vox_per_rgn);
			bspline_interp_pix (dxyz, bxf, p, qidx);
			
			k = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;
			
			for(d=0;d<3;d++) vf[3*k+d] = dxyz[d];
			nv++;
			}	
		}
	}

	/*note that loops go from 1 to roi_dim[]-1 for gradient calculation*/
	for (rk = 1; rk < bxf->roi_dim[2]-1; rk++) {
	p[2] = rk / bxf->vox_per_rgn[2];
	q[2] = rk % bxf->vox_per_rgn[2];
	for (rj = 1; rj < bxf->roi_dim[1]-1; rj++) {
	    p[1] = rj / bxf->vox_per_rgn[1];
	    q[1] = rj % bxf->vox_per_rgn[1];
	    for (ri = 1; ri < bxf->roi_dim[0]-1; ri++) {
			p[0] = ri / bxf->vox_per_rgn[0];
			q[0] = ri % bxf->vox_per_rgn[0];

			rvec[0]=ri; rvec[1]=rj; rvec[2]=rk;

//			for(d=0;d<3;d++)
//				dux_dx[d] = bspline_gradient_1st_derivative(d, rvec, bxf->img_spacing, vf, fixed->dim);

//			for(d=0;d<3;d++) grad_score += (dux_dx[d]*dux_dx[d]);
            num_vox++;

			for(k=0;k<3;k++)
			for(d1=0;d1<3;d1++)
			for(d2=0;d2<3;d2++)
			{
			du = bspline_gradient_2nd_derivative( k, d1, d2, rvec, bxf->img_spacing, vf, fixed->dim);
			grad_score += (du*du); 
			}


			/* updating gradient */
	
//		    for(d=0;d<3;d++) dc_dv[d] = 0;

//			for(k=0;k<3;k++)
//			for(d=0;d<3;d++)
//			dc_dv[k] += dux_dx[d] * bspline_gradient_2nd_derivative( k, d, d, rvec, bxf->img_spacing, vf, fixed->dim);

//			for(d=0;d<3;d++)
//			dc_dv[d] *= (parms->young_modulus/nv);
			
	//		bspline_update_grad (bst, bxf, p, qidx, dc_dv);
			}
		}
	}

	free(vf);

	interval = plm_timer_report (&timer);
	grad_score *= (parms->young_modulus / num_vox);
	printf ("        GRAD_COST %.4f     [%.3f secs]\n", grad_score, interval);
    ssd->score += grad_score;
}
