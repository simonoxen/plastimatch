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
#include "bspline_opts.h"
#include "bspline_regularize_numeric.h"
#include "logfile.h"
#include "math_util.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume_macros.h"

/* Prototypes */
static void bspline_xform_create_qlut_grad (
    Bspline_xform* bxf, float img_spacing[3], int vox_per_rgn[3]);
static void bspline_xform_free_qlut_grad (Bspline_xform* bxf);

/* Flavor 'a' */
float
vf_regularize_numerical (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf,
    const Volume* vol
)
{
#if defined (DEBUG)
    FILE* fp[3];
#endif

    int i,j,k,c;
    float *img = (float*) vol->img;

    float dx = vol->spacing[0];
    float dy = vol->spacing[1];
    float dz = vol->spacing[2];

    float inv_dxdx = 1.0f / (dx * dx);
    float inv_dydy = 1.0f / (dy * dy);
    float inv_dzdz = 1.0f / (dz * dz);

    float inv_dxdy = 0.25f / (dx*dy);
    float inv_dxdz = 0.25f / (dx*dz);
    float inv_dydz = 0.25f / (dy*dz);

    /* LUT indices */
    int p[3], q[3];

    /* Index of current point-of-interest (POI) */
    int idx_poi;

    /* Indices of POI's SxS neighbors */
    int idx_in, idx_ip;
    int idx_jn, idx_jp;
    int idx_kn, idx_kp;

    /* Indicies of POI's diagonal neighbors */
    int idx_injn, idx_injp, idx_ipjn, idx_ipjp;
    int idx_inkn, idx_inkp, idx_ipkn, idx_ipkp;
    int idx_jnkn, idx_jnkp, idx_jpkn, idx_jpkp;

    /* Deformation vector @ POI */
    float *vec_poi;

    /* Vectors of POI's SxS neighbors */
    float *vec_in, *vec_ip;
    float *vec_jn, *vec_jp;
    float *vec_kn, *vec_kp;

    /* Vectors of POI's diagonal neighbors */
    float *vec_injn, *vec_injp;
    float *vec_ipjn, *vec_ipjp;
    float *vec_inkn, *vec_inkp;
    float *vec_ipkn, *vec_ipkp;
    float *vec_jnkn, *vec_jnkp;
    float *vec_jpkn, *vec_jpkp;

    /* Vector's partial spatial derivatives */
    float d2_dx2[3],  d2_dxdy[3];
    float d2_dy2[3],  d2_dxdz[3];
    float d2_dz2[3],  d2_dydz[3];

    /* Square of 2nd derivative */
    float d2_sq, dd2_dxdy;

    /* Smoothness */
    float S, SS;

#if defined (DEBUG)
    printf ("Warning: compiled with DEBUG : writing to to files:\n");
    printf ("  d2ux_dxy_sq.txt\n"); fp[0] = fopen ("d2ux_dxdy_sq.txt", "w");
    printf ("  d2uy_dxy_sq.txt\n"); fp[1] = fopen ("d2uy_dxdy_sq.txt", "w");
    printf ("  d2uz_dxy_sq.txt\n"); fp[2] = fopen ("d2uz_dxdy_sq.txt", "w");
#endif

    S = 0.0f, SS=0.0f;
    for (k = 1; k < vol->dim[2]-1; k++) {
	p[2] = k / bxf->vox_per_rgn[2];
	q[2] = k % bxf->vox_per_rgn[2];
        for (j = 1; j < vol->dim[1]-1; j++) {
	    p[1] = j / bxf->vox_per_rgn[1];
	    q[1] = j % bxf->vox_per_rgn[1];
            for (i = 1; i < vol->dim[0]-1; i++) {
		p[0] = i / bxf->vox_per_rgn[0];
		q[0] = i % bxf->vox_per_rgn[0];

                /* Load indicies relevant to current POI */
                idx_poi = volume_index (vol->dim, i, j, k);

                idx_in = volume_index (vol->dim, i-1  , j,   k);
                idx_ip = volume_index (vol->dim, i+1,   j,   k);
                idx_jn = volume_index (vol->dim,   i, j-1,   k);
                idx_jp = volume_index (vol->dim,   i, j+1,   k);
                idx_kn = volume_index (vol->dim,   i,   j, k-1);
                idx_kp = volume_index (vol->dim,   i,   j, k+1);

                idx_injn = volume_index (vol->dim, i-1, j-1,   k);
                idx_injp = volume_index (vol->dim, i-1, j+1,   k);
                idx_ipjn = volume_index (vol->dim, i+1, j-1,   k);
                idx_ipjp = volume_index (vol->dim, i+1, j+1,   k);
                idx_inkn = volume_index (vol->dim, i-1,   j, k-1);
                idx_inkp = volume_index (vol->dim, i-1,   j, k+1);
                idx_ipkn = volume_index (vol->dim, i+1,   j, k-1);
                idx_ipkp = volume_index (vol->dim, i+1,   j, k+1);
                idx_jnkn = volume_index (vol->dim,   i, j-1, k-1);
                idx_jnkp = volume_index (vol->dim,   i, j-1, k+1);
                idx_jpkn = volume_index (vol->dim,   i, j+1, k-1);
                idx_jpkp = volume_index (vol->dim,   i, j+1, k+1);

                /* Load vectors relevant to current POI */
                vec_poi = &img[3*idx_poi];

                vec_in = &img[3*idx_in]; vec_ip = &img[3*idx_ip];
                vec_jn = &img[3*idx_jn]; vec_jp = &img[3*idx_jp];
                vec_kn = &img[3*idx_kn]; vec_kp = &img[3*idx_kp];

                vec_injn = &img[3*idx_injn]; vec_injp = &img[3*idx_injp];
                vec_ipjn = &img[3*idx_ipjn]; vec_ipjp = &img[3*idx_ipjp];
                vec_inkn = &img[3*idx_inkn]; vec_inkp = &img[3*idx_inkp];
                vec_ipkn = &img[3*idx_ipkn]; vec_ipkp = &img[3*idx_ipkp];
                vec_jnkn = &img[3*idx_jnkn]; vec_jnkp = &img[3*idx_jnkp];
                vec_jpkn = &img[3*idx_jpkn]; vec_jpkp = &img[3*idx_jpkp];

                /* Compute components */
                d2_sq = 0.0f, dd2_dxdy=0.0f;
                for (c=0; c<3; c++) {
                    d2_dx2[c] = inv_dxdx 
			* (vec_ip[c] - 2.0f*vec_poi[c] + vec_in[c]);
                    d2_dy2[c] = inv_dydy 
			* (vec_jp[c] - 2.0f*vec_poi[c] + vec_jn[c]);
                    d2_dz2[c] = inv_dzdz 
			* (vec_kp[c] - 2.0f*vec_poi[c] + vec_kn[c]);

                    d2_dxdy[c] = inv_dxdy * (
                        vec_injn[c] - vec_injp[c] - vec_ipjn[c] + vec_ipjp[c]);
                    d2_dxdz[c] = inv_dxdz * (
                        vec_inkn[c] - vec_inkp[c] - vec_ipkn[c] + vec_ipkp[c]);
                    d2_dydz[c] = inv_dydz * (
                        vec_jnkn[c] - vec_jnkp[c] - vec_jpkn[c] + vec_jpkp[c]);

		    /* Accumulate score for this component, for this voxel */
                    d2_sq += 
			d2_dx2[c]*d2_dx2[c] + 
			d2_dy2[c]*d2_dy2[c] +
			d2_dz2[c]*d2_dz2[c] + 
			2.0f * (
			    d2_dxdy[c]*d2_dxdy[c] +
			    d2_dxdz[c]*d2_dxdz[c] +
			    d2_dydz[c]*d2_dydz[c]
                        );
#if defined (DEBUG)
                    fprintf (fp[c], "(%i,%i,%i) : %15e\n", 
			i,j,k, (d2_dxdy[c]*d2_dxdy[c]));
#endif
		    /* Accumulate grad for this component, for this voxel */
		    int pidx = volume_index (bxf->rdims, p);
		    int qidx = volume_index (bxf->vox_per_rgn, q);
#if defined (commentout)
		    bspline_update_grad_b (bst, bxf, pidx, qidx, dc_dv);
#endif
                }
                S += d2_sq;
            }
        }
    }

    /* Integrate */
    S *= dx*dy*dz;

#if defined (DEBUG)
    for (i=0; i<3; i++) {
        fclose(fp[i]);
    }
#endif

    return S;
}

void
bspline_regularize_numeric_a (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf
)
{
    Volume *vf = bspline_compute_vf (bxf);

    float S = vf_regularize_numerical (ssd, parms, rst, bxf, vf);

    delete vf;
}

void
bspline_regularize_numeric_a_init (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
)
{
}

void
bspline_regularize_numeric_a_destroy (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
)
{
}

/* Flavor 'd' */
static
void bspline_xform_create_qlut_grad 
(
    Bspline_xform* bxf,         /* Output: bxf with new LUTs */
    float img_spacing[3],       /* Image spacing (in mm) */
    int vox_per_rgn[3])         /* Knot spacing (in vox) */
{
    int i, j, k, p;
    int tx, ty, tz;
    float *A, *B, *C;
    float *Ax, *By, *Cz, *Axx, *Byy, *Czz;
    int q_lut_size;

    q_lut_size = sizeof(float) * bxf->vox_per_rgn[0] 
	* bxf->vox_per_rgn[1] 
	* bxf->vox_per_rgn[2] 
	* 64;
    logfile_printf("Creating gradient multiplier LUTs, %d bytes each\n", q_lut_size);

    bxf->q_dxdyz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_dxdyz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");
	
    bxf->q_xdydz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_xdydz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    bxf->q_dxydz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_dxydz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");
	
    bxf->q_d2xyz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_d2xyz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    bxf->q_xd2yz_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_xd2yz_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    bxf->q_xyd2z_lut = (float*) malloc ( q_lut_size );
    if (!bxf->q_xyd2z_lut) print_and_exit ("Error allocating memory for q_grad_lut\n");

    A = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    B = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    C = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    Ax = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    By = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    Cz = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    Axx = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    Byy = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    Czz = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);

    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
	float ii = ((float) i) / bxf->vox_per_rgn[0];
	float t3 = ii*ii*ii;
	float t2 = ii*ii;
	float t1 = ii;
	A[i*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	A[i*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	A[i*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	A[i*4+3] = (1.0/6.0) * (+ 1.0 * t3);

	Ax[i*4+0] =(1.0/6.0) * (- 3.0 * t2 + 6.0 * t1 - 3.0           );
	Ax[i*4+1] =(1.0/6.0) * (+ 9.0 * t2 - 12.0* t1                 );
	Ax[i*4+2] =(1.0/6.0) * (- 9.0 * t2 + 6.0 * t1 + 3.0           );
	Ax[i*4+3] =(1.0/6.0) * (+ 3.0 * t2);

	Axx[i*4+0]=(1.0/6.0) * (- 6.0 * t1 + 6.0                     );
	Axx[i*4+1]=(1.0/6.0) * (+18.0 * t1 - 12.0                    );
	Axx[i*4+2]=(1.0/6.0) * (-18.0 * t1 + 6.0                     );
	Axx[i*4+3]=(1.0/6.0) * (+ 6.0 * t1);
    }
    for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
	float jj = ((float) j) / bxf->vox_per_rgn[1];
	float t3 = jj*jj*jj;
	float t2 = jj*jj;
	float t1 = jj;
	B[j*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	B[j*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	B[j*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	B[j*4+3] = (1.0/6.0) * (+ 1.0 * t3);

	By[j*4+0] =(1.0/6.0) * (- 3.0 * t2 + 6.0 * t1 - 3.0           );
	By[j*4+1] =(1.0/6.0) * (+ 9.0 * t2 - 12.0* t1                 );
	By[j*4+2] =(1.0/6.0) * (- 9.0 * t2 + 6.0 * t1 + 3.0           );
	By[j*4+3] =(1.0/6.0) * (+ 3.0 * t2);

	Byy[j*4+0]=(1.0/6.0) * (- 6.0 * t1 + 6.0                     );
	Byy[j*4+1]=(1.0/6.0) * (+18.0 * t1 - 12.0                    );
	Byy[j*4+2]=(1.0/6.0) * (-18.0 * t1 + 6.0                     );
	Byy[j*4+3]=(1.0/6.0) * (+ 6.0 * t1);
    }
    for (k = 0; k < bxf->vox_per_rgn[2]; k++) {
	float kk = ((float) k) / bxf->vox_per_rgn[2];
	float t3 = kk*kk*kk;
	float t2 = kk*kk;
	float t1 = kk;
	C[k*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	C[k*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	C[k*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	C[k*4+3] = (1.0/6.0) * (+ 1.0 * t3);

	Cz[k*4+0] =(1.0/6.0) * (- 3.0 * t2 + 6.0 * t1 - 3.0           );
	Cz[k*4+1] =(1.0/6.0) * (+ 9.0 * t2 - 12.0* t1                 );
	Cz[k*4+2] =(1.0/6.0) * (- 9.0 * t2 + 6.0 * t1 + 3.0           );
	Cz[k*4+3] =(1.0/6.0) * (+ 3.0 * t2);

	Czz[k*4+0]=(1.0/6.0) * (- 6.0 * t1 + 6.0                     );
	Czz[k*4+1]=(1.0/6.0) * (+18.0 * t1 - 12.0                    );
	Czz[k*4+2]=(1.0/6.0) * (-18.0 * t1 + 6.0                     );
	Czz[k*4+3]=(1.0/6.0) * (+ 6.0 * t1);
    }

    p = 0;
    for (k = 0; k < bxf->vox_per_rgn[2]; k++) {
	for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
	    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
				
			    bxf->q_dxdyz_lut[p] = Ax[i*4+tx] * By[j*4+ty] * C[k*4+tz];
			    bxf->q_xdydz_lut[p] = A[i*4+tx] * By[j*4+ty] * Cz[k*4+tz];
			    bxf->q_dxydz_lut[p] = Ax[i*4+tx] * B[j*4+ty] * Cz[k*4+tz];

			    bxf->q_d2xyz_lut[p] = Axx[i*4+tx] * B[j*4+ty] * C[k*4+tz];
			    bxf->q_xd2yz_lut[p] = A[i*4+tx] * Byy[j*4+ty] * C[k*4+tz];
			    bxf->q_xyd2z_lut[p] = A[i*4+tx] * B[j*4+ty] * Czz[k*4+tz];

			    p++;
			}
		    }
		}
	    }
	}
    }
    free (C);
    free (B);
    free (A);
    free (Ax); free(By); free(Cz); free(Axx); free(Byy); free(Czz);
}

static
void
bspline_xform_free_qlut_grad (Bspline_xform* bxf)
{
    free (bxf->q_dxdyz_lut);
    free (bxf->q_dxydz_lut);
    free (bxf->q_xdydz_lut);
    free (bxf->q_d2xyz_lut);
    free (bxf->q_xd2yz_lut);
    free (bxf->q_xyd2z_lut);
}



#define USE_FAST_CODE 1

/*internal use only - get k-th component (k=0,1,2) of 
the vector field at location r using pre-rendered vf */
static float
f2 (int k, int r[3], float  *vf, int *dims)
{
    int d;
    d = r[2] * dims[0] * dims[1] + r[1] * dims[0] + r[0];
    return vf[3*d+k];
}

/*
first derivative of the vector field du_i/dx_i using pre-rendered vf
calculated at position r[3] = (ri rj rk) in voxels */
float
bspline_regularize_1st_derivative (
    int i,
    int r[3],
    float h[3],
    float *vf,
    int *dims
)
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
float
bspline_regularize_2nd_derivative (
    int k,
    int i,
    int j,
    int r[3],
    float h[3],
    float *vf,
    int *dims
)
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
    const Bspline_xform* bxf, 
    int p[3], 
    int qidx, 
    int derive1, 
    int derive2
)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = 0;

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
    const Bspline_xform* bxf, 
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
    Bspline_score *ssd, 
    const Bspline_xform* bxf, 
    int p[3], 
    int qidx, 
    float dc_dv[3], 
    int derive1, 
    int derive2
)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = 0;

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
    Bspline_score *ssd, 
    const Bspline_xform* bxf, 
    int p[3], 
    int qidx, 
    float dc_dv[3], 
    float *q_lut
)
{
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

#if defined (USE_FAST_CODE)
static double
update_score_and_grad (
    Bspline_score *ssd, 
    const Bspline_xform* bxf, 
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

    bspline_regularize_hessian_update_grad_b (ssd, bxf, p, qidx, dc_dv, qlut);

    return score;
}
#endif

void
bspline_regularize_numeric_d (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf
)
{
    double grad_score;
    int ri, rj, rk;
    int fi, fj, fk;
    int p[3];
    int q[3];
    int qidx;
    int num_vox;
    Plm_timer timer;
    double interval;
    float grad_coeff;
    float raw_score;

    grad_score = 0;
    num_vox = bxf->roi_dim[0] * bxf->roi_dim[1] * bxf->roi_dim[2];
    grad_coeff = parms->lambda / num_vox;

    plm_timer_start (&timer);

    printf("---- YOUNG MODULUS %f\n", parms->lambda);

    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
	p[2] = rk / bxf->vox_per_rgn[2];
	q[2] = rk % bxf->vox_per_rgn[2];
	for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
	    p[1] = rj / bxf->vox_per_rgn[1];
	    q[1] = rj % bxf->vox_per_rgn[1];
	    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
		p[0] = ri / bxf->vox_per_rgn[0];
		q[0] = ri % bxf->vox_per_rgn[0];
			
		qidx = volume_index (bxf->vox_per_rgn, q);
#if defined (USE_FAST_CODE)
		grad_score += update_score_and_grad (
		    ssd, bxf, p, qidx, grad_coeff, 1,
		    &bxf->q_d2xyz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    ssd, bxf, p, qidx, grad_coeff, 1,
		    &bxf->q_xd2yz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    ssd, bxf, p, qidx, grad_coeff, 1,
		    &bxf->q_xyd2z_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    ssd, bxf, p, qidx, grad_coeff, 2,
		    &bxf->q_dxdyz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    ssd, bxf, p, qidx, grad_coeff, 2,
		    &bxf->q_dxydz_lut[qidx*64]);
		grad_score += update_score_and_grad (
		    ssd, bxf, p, qidx, grad_coeff, 2,
		    &bxf->q_xdydz_lut[qidx*64]);
#else
		for (int d1=0;d1<3;d1++) {
		    for (int d2=d1;d2<3;d2++) { //six different components only
			float dxyz[3];
			float dc_dv[3];
			bspline_regularize_hessian_component (
			    dxyz, bxf, p, qidx, d1, d2);

			/* Note: dxyz[i] = du_i/(dx_d1 dx_d2) */
			float weight;
			if (d1!=d2) weight = 2 ; else weight = 1;
			for (int d3=0;d3<3;d3++) {
			    grad_score += weight*(dxyz[d3]*dxyz[d3]);
			}	
			dc_dv[0] = weight * grad_coeff * dxyz[0];
			dc_dv[1] = weight * grad_coeff * dxyz[1];
			dc_dv[2] = weight * grad_coeff * dxyz[2];

			bspline_regularize_hessian_update_grad (
			    ssd, bxf, p, qidx, dc_dv, d1, d2);
		    }
		}
#endif
	    }
	}

	interval = plm_timer_report (&timer);
	raw_score = grad_score /num_vox;
	grad_score *= (parms->lambda / num_vox);
	printf ("        GRAD_COST %.4f   RAW_GRAD %.4f   [%.3f secs]\n", grad_score, raw_score, interval);
	ssd->score += grad_score;
    }
    printf ("SCORE=%.4f\n", ssd->score);
}

void
bspline_regularize_numeric_d_init (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
)
{
    bspline_xform_create_qlut_grad (bxf, bxf->img_spacing, bxf->vox_per_rgn);
}

void
bspline_regularize_numeric_d_destroy (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
)
{
    bspline_xform_free_qlut_grad (bxf);
}
