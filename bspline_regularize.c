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
#include "bspline_regularize.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "timer.h"

//#define USE_FAST_CODE 1

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
float bspline_regularize_1st_derivative (
    int i,  int r[3], float h[3], float *vf, int *dims )
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
float bspline_regularize_2nd_derivative( int k, int i, int j,  int r[3], float h[3], float *vf, int *dims )
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

/* out[i] = 2nd derivative of i-th component (i=0,1,2<->x,y,z) of the vector field
with respect to variables derive1 and derive2 (0,1,2<->x,y,z),
or (derive1,derive2)th element of the Hessian of the i-th component of the vector field 
*/
void
bspline_regularize_hessian_component (
    float out[3], 
    BSPLINE_Xform* bxf, 
    int p[3], 
    int qidx, 
    int derive1, 
    int derive2
)
{
    int i, j, k, m;
    int cidx;
    float* q_lut;

    if (derive1==0 && derive2==0) q_lut = &bxf->q_d2xyz_lut[qidx*64];
    if (derive1==1 && derive2==1) q_lut = &bxf->q_xd2yz_lut[qidx*64];
    if (derive1==2 && derive2==2) q_lut = &bxf->q_xyd2z_lut[qidx*64];

    if (derive1==0 && derive2==1) q_lut = &bxf->q_dxdyz_lut[qidx*64];
    if (derive1==1 && derive2==0) q_lut = &bxf->q_dxdyz_lut[qidx*64];
	
    if (derive1==0 && derive2==2) q_lut = &bxf->q_dxydz_lut[qidx*64];
    if (derive1==2 && derive2==0) q_lut = &bxf->q_dxydz_lut[qidx*64];

    if (derive1==1 && derive2==2) q_lut = &bxf->q_xdydz_lut[qidx*64];
    if (derive1==2 && derive2==1) q_lut = &bxf->q_xdydz_lut[qidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
		    + (p[1] + j) * bxf->cdims[0]
		    + (p[0] + i);
		cidx = cidx * 3;
		out[0] += q_lut[m] * bxf->coeff[cidx+0] ;
		out[1] += q_lut[m] * bxf->coeff[cidx+1] ;
		out[2] += q_lut[m] * bxf->coeff[cidx+2] ;
		m ++;
	    }
	}
    }
}

void
bspline_regularize_hessian_component_b (
    float out[3], 
    BSPLINE_Xform* bxf, 
    int p[3], 
    int qidx, 
    float *q_lut
)
{
    int i, j, k, m;
    int cidx;

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
		    + (p[1] + j) * bxf->cdims[0]
		    + (p[0] + i);
		cidx = cidx * 3;
		out[0] += q_lut[m] * bxf->coeff[cidx+0];
		out[1] += q_lut[m] * bxf->coeff[cidx+1];
		out[2] += q_lut[m] * bxf->coeff[cidx+2];
		m ++;
	    }
	}
    }
}


void
bspline_regularize_hessian_update_grad (
    Bspline_state *bst, 
    BSPLINE_Xform* bxf, 
    int p[3], 
    int qidx, 
    float dc_dv[3], 
    int derive1, 
    int derive2
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i, j, k, m;
    int cidx;
    float* q_lut;

    if (derive1==0 && derive2==0) q_lut = &bxf->q_d2xyz_lut[qidx*64];
    if (derive1==1 && derive2==1) q_lut = &bxf->q_xd2yz_lut[qidx*64];
    if (derive1==2 && derive2==2) q_lut = &bxf->q_xyd2z_lut[qidx*64];

    if (derive1==0 && derive2==1) q_lut = &bxf->q_dxdyz_lut[qidx*64];
    if (derive1==1 && derive2==0) q_lut = &bxf->q_dxdyz_lut[qidx*64];
	
    if (derive1==0 && derive2==2) q_lut = &bxf->q_dxydz_lut[qidx*64];
    if (derive1==2 && derive2==0) q_lut = &bxf->q_dxydz_lut[qidx*64];

    if (derive1==1 && derive2==2) q_lut = &bxf->q_xdydz_lut[qidx*64];
    if (derive1==2 && derive2==1) q_lut = &bxf->q_xdydz_lut[qidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
		    + (p[1] + j) * bxf->cdims[0]
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


void
bspline_regularize_hessian_update_grad_b (
    Bspline_state *bst, 
    BSPLINE_Xform* bxf, 
    int p[3], 
    int qidx, 
    float dc_dv[3], 
    float *q_lut
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i, j, k, m;
    int cidx;

    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
		    + (p[1] + j) * bxf->cdims[0]
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


void
bspline_regularize_score_from_prerendered (
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
    //float dc_dv[3];
    //float dux_dx[3];
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
		//				dux_dx[d] = bspline_regularize_1st_derivative(d, rvec, bxf->img_spacing, vf, fixed->dim);

		//			for(d=0;d<3;d++) grad_score += (dux_dx[d]*dux_dx[d]);
		num_vox++;

		for(k=0;k<3;k++)
		    for(d1=0;d1<3;d1++)
			for(d2=0;d2<3;d2++)
			{
			    du = bspline_regularize_2nd_derivative( k, d1, d2, rvec, bxf->img_spacing, vf, fixed->dim);
			    grad_score += (du*du); 
			}


		/* updating gradient */
	
		//		    for(d=0;d<3;d++) dc_dv[d] = 0;

		//			for(k=0;k<3;k++)
		//			for(d=0;d<3;d++)
		//			dc_dv[k] += dux_dx[d] * bspline_regularize_2nd_derivative( k, d, d, rvec, bxf->img_spacing, vf, fixed->dim);

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

#if defined (USE_FAST_CODE)
static double
update_score_and_grad (
    Bspline_state *bst, 
    BSPLINE_Xform* bxf, 
    int p[3], 
    int qidx, 
    float grad_coeff, 
    float weight, // 2 or 1 for cross/non-cross derivatives
    float *qlut
)
{
    int d3;
    float dxyz[3];
    float dc_dv[3];
    double score = 0.0;

    bspline_regularize_hessian_component_b (dxyz, bxf, p, qidx, qlut);

    for (d3=0; d3<3; d3++) {
	score += weight * (dxyz[d3]*dxyz[d3]);
	dc_dv[d3] = weight * grad_coeff * dxyz[d3];
    }

    bspline_regularize_hessian_update_grad_b (bst, bxf, p, qidx, dc_dv, qlut);

    return score;
}
#endif

void
bspline_regularize_score (
    BSPLINE_Parms *parms, 
    Bspline_state *bst, 
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    double grad_score;
    int ri, rj, rk;
    int fi, fj, fk;
    int p[3];
    int q[3];
    float dxyz[3];
    int qidx;
    float dc_dv[3];
    int num_vox;
    int d1,d2,d3;
    Timer timer;
    double interval;
    float grad_coeff;
    float raw_score;
	float weight;

    grad_score = 0;
    num_vox = bxf->roi_dim[0] * bxf->roi_dim[1] * bxf->roi_dim[2];
    grad_coeff = parms->young_modulus / num_vox;

    plm_timer_start (&timer);

    printf("---- YOUNG MODULUS %f\n", parms->young_modulus);

    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
	p[2] = rk / bxf->vox_per_rgn[2];
	q[2] = rk % bxf->vox_per_rgn[2];
	for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
	    p[1] = rj / bxf->vox_per_rgn[1];
	    q[1] = rj % bxf->vox_per_rgn[1];
	    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
		p[0] = ri / bxf->vox_per_rgn[0];
		q[0] = ri % bxf->vox_per_rgn[0];
			
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
#if defined (USE_FAST_CODE)
		grad_score += update_score_and_grad (
		    bst, bxf, p, qidx, grad_coeff, 1,
		    &bxf->q_d2xyz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    bst, bxf, p, qidx, grad_coeff, 1,
		    &bxf->q_xd2yz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    bst, bxf, p, qidx, grad_coeff, 1,
		    &bxf->q_xyd2z_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    bst, bxf, p, qidx, grad_coeff, 2,
		    &bxf->q_dxdyz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    bst, bxf, p, qidx, grad_coeff, 2,
		    &bxf->q_dxydz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    bst, bxf, p, qidx, grad_coeff, 2,
		    &bxf->q_xdydz_lut[qidx*64]);
#else
		for (d1=0;d1<3;d1++) {
		    for (d2=d1;d2<3;d2++) { //six different components only
			bspline_regularize_hessian_component (
			    dxyz, bxf, p, qidx, d1, d2);
			//dxyz[i] = du_i/(dx_d1 dx_d2)
			if (d1!=d2) weight = 2 ; else weight = 1;
			for(d3=0;d3<3;d3++) grad_score += weight*(dxyz[d3]*dxyz[d3]);
	
			dc_dv[0] = weight * grad_coeff * dxyz[0];
			dc_dv[1] = weight * grad_coeff * dxyz[1];
			dc_dv[2] = weight * grad_coeff * dxyz[2];

			bspline_regularize_hessian_update_grad (
			    bst, bxf, p, qidx, dc_dv, d1, d2);
		    }
		}

#endif
	    }
	}

	interval = plm_timer_report (&timer);
	raw_score = grad_score /num_vox;
	grad_score *= (parms->young_modulus / num_vox);
	printf ("        GRAD_COST %.4f   RAW_GRAD %.4f   [%.3f secs]\n", grad_score, raw_score, interval);
	ssd->score += grad_score;
    }
}
