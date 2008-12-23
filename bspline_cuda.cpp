/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "volume.h"
#include "readmha.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_cuda.h"

inline void bspline_update_grad_b_inline (
	BSPLINE_Parms* parms, 
	BSPLINE_Xform* bxf, 
	int pidx, 
	int qidx, 
	float dc_dv[3])
{
    BSPLINE_Score* ssd = &parms->ssd;
    int i, j, k, m;
    int cidx;
    float* q_lut = &bxf->q_lut[qidx*64];
    int* c_lut = &bxf->c_lut[pidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
		for (j = 0; j < 4; j++) {
			for (i = 0; i < 4; i++) {
				cidx = 3 * c_lut[m];
				ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
				ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
				ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
				m++;
			}
		}
    }
}

void bspline_cuda_score_mse(
	BSPLINE_Parms *parms, 
	BSPLINE_Xform* bxf, 
	Volume *fixed, 
	Volume *moving, 
	Volume *moving_grad)
{
    BSPLINE_Score* ssd = &parms->ssd;
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
    float diff;
    float dc_dv[3];
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;
    int dd = 0;

	/* BEGIN CUDA VARIABLES */
	float *host_diff;
	float *host_dc_dv_x;
	float *host_dc_dv_y;
	float *host_dc_dv_z;
	float *host_score;
	float *host_grad_norm;
	float *host_grad_mean;
	int diff_errors = 0;
	int dc_dv_errors = 0;
	/* END CUDA VARIBLES */

    if (parms->debug) {
	sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    start_clock = clock();

	/* Run CUDA code now so the results can be compared below. */
	/* BEGIN CUDA CALLS */
	host_diff = (float*)malloc(fixed->npix * sizeof(float));
	host_dc_dv_x = (float*)malloc(fixed->npix * sizeof(float));
	host_dc_dv_y = (float*)malloc(fixed->npix * sizeof(float));
	host_dc_dv_z = (float*)malloc(fixed->npix * sizeof(float));
	host_score   = (float*)malloc(sizeof(float));
	host_grad_norm = (float*)malloc(sizeof(float));
	host_grad_mean = (float*)malloc(sizeof(float));

	bspline_cuda_run_kernels(
		fixed,
		moving,
		moving_grad,
		bxf,
		parms,
		host_diff,
		host_dc_dv_x,
		host_dc_dv_y,
		host_dc_dv_z,
		host_score);
	/* END CUDA CALLS */

    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
		p[2] = rk / bxf->vox_per_rgn[2];
		q[2] = rk % bxf->vox_per_rgn[2];
		fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
		
		for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
			p[1] = rj / bxf->vox_per_rgn[1];
			q[1] = rj % bxf->vox_per_rgn[1];
			fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
		    
			for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
				p[0] = ri / bxf->vox_per_rgn[0];
				q[0] = ri % bxf->vox_per_rgn[0];
				fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;

				// Get B-spline deformation vector.
				pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
				qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];

				// Compute coordinate of fixed image voxel.
				fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

				/*
				bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

				// Compare dxyz result to the one computed by CUDA.
				if(((host_dx[fv] < (dxyz[0] - 0.0001)) || (host_dx[fv] > (dxyz[0] + 0.0001))) ||
					((host_dy[fv] < (dxyz[1] - 0.0001)) || (host_dy[fv] > (dxyz[1] + 0.0001))) ||
					((host_dz[fv] < (dxyz[2] - 0.0001)) || (host_dz[fv] > (dxyz[2] + 0.0001)))) {
					dxyz_errors++;
					if(dxyz_errors < 20) {
						printf("DXYZ ERROR: %d -- CUDA: (%f, %f, %f), CPU: (%f, %f, %f)\n", 
							fv, 
							host_dx[fv], 
							host_dy[fv], 
							host_dz[fv], 
							dxyz[0],
							dxyz[1],
							dxyz[2]);
						fflush(stdout);
					}
				}

				// Find correspondence in moving image.
				mx = fx + dxyz[0];
				mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
				if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

				my = fy + dxyz[1];
				mj = (my - moving->offset[1]) / moving->pix_spacing[1];
				if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

				mz = fz + dxyz[2];
				mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
				if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

				// Compute interpolation fractions.
				clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
				clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
				clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

				// Compute moving image intensity using linear interpolation.
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

				// Compute intensity difference.
				diff = f_img[fv] - m_val;

				// Compare diff result to the one computed by CUDA.
				if(((host_diff[fv] < (diff - 0.05)) || (host_diff[fv] > (diff + 0.05)))) {
					diff_errors++;
					if(diff_errors < 20) {
						printf("DIFF ERROR: %d -- CUDA: %f, CPU: %f\n", 
							fv, 
							host_diff[fv], 
							diff);
						fflush(stdout);
					}
				}

				// Compute spatial gradient using nearest neighbors.
				mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;
				dc_dv[0] = diff * m_grad[3*mvr+0];  // x component
				dc_dv[1] = diff * m_grad[3*mvr+1];  // y component
				dc_dv[2] = diff * m_grad[3*mvr+2];  // z component
				
				// Compare dc_dv result to the one computed by CUDA.
				if(((host_dc_dv_x[fv] < (dc_dv[0] - 0.01)) || (host_dc_dv_x[fv] > (dc_dv[0] + 0.01))) ||
					((host_dc_dv_y[fv] < (dc_dv[1] - 0.01)) || (host_dc_dv_y[fv] > (dc_dv[1] + 0.01))) ||
					((host_dc_dv_z[fv] < (dc_dv[2] - 0.01)) || (host_dc_dv_z[fv] > (dc_dv[2] + 0.01)))) {
					dc_dv_errors++;
					if(dc_dv_errors < 20) {
						printf("DC_DV ERROR: %d -- CUDA: (%f, %f, %f), CPU: (%f, %f, %f)\n", 
							fv, 
							host_dc_dv_x[fv], 
							host_dc_dv_y[fv], 
							host_dc_dv_z[fv], 
							dc_dv[0],
							dc_dv[1],
							dc_dv[2]);
						fflush(stdout);
					}
				}
				*/

				diff = host_diff[fv];
				dc_dv[0] = host_dc_dv_x[fv];
				dc_dv[1] = host_dc_dv_y[fv];
				dc_dv[2] = host_dc_dv_z[fv];
				
				bspline_update_grad_b_inline (parms, bxf, pidx, qidx, dc_dv);
				
				// ssd->score += diff * diff;
				// num_vox ++;
			}
		}
    }

    if (parms->debug) {
	fclose (fp);
    }

    //dump_coeff (bxf, "coeff.txt");

    
	bspline_cuda_calculate_gradient(
		parms,
		bxf,
		fixed,
		host_grad_norm,
		host_grad_mean);

	ssd->score = *host_score;
	ssd_grad_norm = *host_grad_norm;
	ssd_grad_mean = *host_grad_mean;

	/* Normalize score for MSE */
	/*
	ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
		ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
		ssd_grad_mean += ssd->grad[i];
		ssd_grad_norm += fabs (ssd->grad[i]);
    }
	*/

    end_clock = clock();

	/*
	printf("%d errors found between CUDA and CPU calculations of dxyz.\n", dxyz_errors);
	printf("%d errors found between CUDA and CPU calculations of diff.\n", diff_errors);
	printf("%d errors found between CUDA and CPU calculations of dc_dv.\n", dc_dv_errors);
	*/

	free(host_diff);
	free(host_dc_dv_x);
	free(host_dc_dv_y);
	free(host_dc_dv_z);
	free(host_score);
	free(host_grad_norm);
	free(host_grad_mean);

    printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}