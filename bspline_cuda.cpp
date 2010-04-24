/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined (_WIN32)
#include <windows.h>
#endif
#include "plm_config.h"
#include "volume.h"
#include "readmha.h"
#include "timer.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "bspline.h"
#include "bspline_cuda.h"

/***********************************************************************
 * A few of the CPU functions are reproduced here for testing purposes.
 * Once the CPU code is removed from the functions below, these
 * functions can be deleted.
 ***********************************************************************/
#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

inline void 
bspline_update_grad_b_inline 
(
 Bspline_state *bst,
 BSPLINE_Xform* bxf, 
 int pidx, 
 int qidx, 
 float dc_dv[3])
{
    BSPLINE_Score* ssd = &bst->ssd;
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

inline void
clamp_linear_interpolate_inline (
    float ma,           // (Unrounded) pixel coordinate (in vox) 
    int dmax,		// Maximum coordinate in this dimension 
    int* maf,		// x, y, or z coord of "floor" pixel in moving img
    int* mar,		// x, y, or z coord of "round" pixel in moving img
    float* fa1,		// Fraction of interpolant for lower index voxel 
    float* fa2		// Fraction of interpolant for upper index voxel
)
{
    float maff = floor(ma);
    *maf = (int) maff;
    *mar = round_int (ma);
    *fa2 = ma - maff;
    if (*maf < 0) {
	*maf = 0;
	*mar = 0;
	*fa2 = 0.0f;
    } else if (*maf >= dmax) {
	*maf = dmax - 1;
	*mar = dmax;
	*fa2 = 1.0f;
    }
    *fa1 = 1.0f - *fa2;
}

inline void
bspline_interp_pix_b_inline (float out[3], BSPLINE_Xform* bxf, int pidx, int qidx)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bxf->q_lut[qidx*64];
    int* c_lut = &bxf->c_lut[pidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = 3 * c_lut[m];
		out[0] += q_lut[m] * bxf->coeff[cidx+0];
		out[1] += q_lut[m] * bxf->coeff[cidx+1];
		out[2] += q_lut[m] * bxf->coeff[cidx+2];
		m ++;
	    }
	}
    }
}

static float
mi_hist_score (BSPLINE_MI_Hist* mi_hist, int num_vox)
{
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;

    int i, j, v;
    float fnv = (float) num_vox;
    float score = 0;
    float hist_thresh = 0.001 / mi_hist->moving.bins / mi_hist->fixed.bins;

    /* Compute cost */
    for (i = 0, v = 0; i < mi_hist->fixed.bins; i++) {
	for (j = 0; j < mi_hist->moving.bins; j++, v++) {
	    if (j_hist[v] > hist_thresh && (j_hist[j] * f_hist[i] > 0)) {
		score -= j_hist[v] * logf (fnv * j_hist[v] / (m_hist[j] * f_hist[i]));
	    }
	}
    }

    score = score / fnv;
    return score;
}

inline void
bspline_mi_hist_lookup (
    long j_idxs[2],		/* Output: Joint histogram indices */
    long m_idxs[2],		/* Output: Moving marginal indices */
    long f_idxs[1],		/* Output: Fixed marginal indices */
    float fxs[2],		/* Output: Fraction contribution at indices */
    BSPLINE_MI_Hist* mi_hist,   /* Input:  The histogram */
    float f_val,		/* Input:  Intensity of fixed image */
    float m_val		        /* Input:  Intensity of moving image */
)
{
    long fl;
    float midx, midx_trunc;
    long ml_1, ml_2;		/* 1-d index of bin 1, bin 2 */
    float mf_1, mf_2;		/* fraction to bin 1, bin 2 */
    long f_idx;	/* Index into 2-d histogram */

    /* Fixed image is binned */
    fl = (long) floor ((f_val - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    f_idx = fl * mi_hist->moving.bins;

    /* This had better not happen! */
    if (fl < 0 || fl >= mi_hist->fixed.bins) {
	fprintf (stderr, "Error: fixed image binning problem.\n"
		 "Bin %ld from val %g parms [off=%g, delt=%g, (%ld bins)]\n",
		 fl, f_val, mi_hist->fixed.offset, mi_hist->fixed.delta,
		 mi_hist->fixed.bins);
	exit (-1);
    }
    
    /* Moving image binning is interpolated (linear, not b-spline) */
    midx = ((m_val - mi_hist->moving.offset) / mi_hist->moving.delta);
    midx_trunc = floorf (midx);
    ml_1 = (long) midx_trunc;
    mf_1 = midx - midx_trunc;    // Always between 0 and 1
    ml_2 = ml_1 + 1;
    mf_2 = 1.0 - mf_1;

    if (ml_1 < 0) {
	/* This had better not happen! */
	fprintf (stderr, "Error: moving image binning problem\n");
	exit (-1);
    } else if (ml_2 >= mi_hist->moving.bins) {
	/* This could happen due to rounding */
	ml_1 = mi_hist->moving.bins - 2;
	ml_2 = mi_hist->moving.bins - 1;
	mf_1 = 0.0;
	mf_2 = 1.0;
    }

    if (mf_1 < 0.0 || mf_1 > 1.0 || mf_2 < 0.0 || mf_2 > 1.0) {
	fprintf (stderr, "Error: MI interpolation problem\n");
	exit (-1);
    }

    j_idxs[0] = f_idx + ml_1;
    j_idxs[1] = f_idx + ml_2;
    fxs[0] = mf_1;
    fxs[1] = mf_2;
    f_idxs[0] = fl;
    m_idxs[0] = ml_1;
    m_idxs[1] = ml_2;
}

inline void
clamp_quadratic_interpolate_grad_inline (
    float ma,           /* (Unrounded) pixel coordinate (in vox units) */
    long dmax,		/* Maximum coordinate in this dimension */
    long maqs[3],	/* x, y, or z coord of 3 pixels in moving img */
    float faqs[3]	/* Gradient interpolant for 3 voxels */
)
{
    float marf = floorf (ma + 0.5);	/* marf = ma, rounded, floating */
    long mari = (long) marf;		/* mari = ma, rounded, integer */

    float t = ma - marf + 0.5;

    faqs[0] = -1.0f + t;
    faqs[1] = -2.0f * t + 1.0f;
    faqs[2] = t;

    maqs[0] = mari - 1;
    maqs[1] = mari;
    maqs[2] = mari + 1;

    if (maqs[0] < 0) {
	faqs[0] = faqs[1] = faqs[2] = 0.0f;	/* No gradient at image boundary */
	maqs[0] = 0;
	if (maqs[1] < 0) {
	    maqs[1] = 0;
	    if (maqs[2] < 0) {
		maqs[2] = 0;
	    }
	}
    } else if (maqs[2] >= dmax) {
	faqs[0] = faqs[1] = faqs[2] = 0.0f;	/* No gradient at image boundary */
	maqs[2] = dmax - 1;
	if (maqs[1] >= dmax) {
	    maqs[1] = dmax - 1;
	    if (maqs[0] >= dmax) {
		maqs[0] = dmax - 1;
	    }
	}
    }
}

inline float
compute_dS_dP (
    float* j_hist, 
    float* f_hist, 
    float* m_hist, 
    long* j_idxs, 
    long* f_idxs, 
    long* m_idxs, 
    float num_vox_f, 
    float* fxs, 
    float score, 
    int debug
)
{
    float dS_dP_0, dS_dP_1, dS_dP;
    const float j_hist_thresh = 0.0001f;

    if (debug) {
	fprintf (stderr, "j=[%ld %ld] (%g %g), "
		 "f=[%ld] (%g), "
		 "m=[%ld %ld] (%g %g), "
		 "fxs = (%g %g)\n",
		 j_idxs[0], j_idxs[1], j_hist[j_idxs[0]], j_hist[j_idxs[1]],
		 f_idxs[0], f_hist[f_idxs[0]],
		 m_idxs[0], m_idxs[1], m_hist[m_idxs[0]], m_hist[m_idxs[1]],
		 fxs[0], fxs[1]);
    }

    if (j_hist[j_idxs[0]] < j_hist_thresh) {
	dS_dP_0 = 0.0f;
    } else {
	dS_dP_0 = fxs[0] * (logf((num_vox_f * j_hist[j_idxs[0]]) / (f_hist[f_idxs[0]] * m_hist[m_idxs[0]])) - score);
    }
    if (j_hist[j_idxs[1]] < j_hist_thresh) {
	dS_dP_1 = 0.0f;
    } else {
	dS_dP_1 = fxs[1] * (logf((num_vox_f * j_hist[j_idxs[1]]) / (f_hist[f_idxs[0]] * m_hist[m_idxs[1]])) - score);
    }

    dS_dP = dS_dP_0 + dS_dP_1;
    if (debug) {
	fprintf (stderr, "dS_dP %g = %g %g\n", dS_dP, dS_dP_0, dS_dP_1);
    }

    return dS_dP;
}





////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_MI_a (
    BSPLINE_Parms *parms,
    Bspline_state *bst,
    BSPLINE_Xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad,
    Dev_Pointers_Bspline *dev_ptrs)
{

    // --- DECLARE LOCAL VARIABLES ------------------------------
    BSPLINE_Score* ssd;	// Holds the SSD "Score" information
    int num_vox;		// Holds # of voxels in the fixed volume
    Timer timer;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;

    static int it=0;	// Holds Iteration Number
    char debug_fn[1024];	// Debug message buffer
    FILE* fp = NULL;	// File Pointer to Debug File
    //	int i;
    // ----------------------------------------------------------

    // --- TEMP CPU CODE VARIABLES ------------------------------
    int ri, rj, rk;
    int fi, fj, fk, fv;
    float mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    long miqs[3], mjqs[3], mkqs[3];
    float fxqs[3], fyqs[3], fzqs[3];
    int p[3];
    int q[3];
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float dxyz[3];
    float num_vox_f;
    int pidx, qidx;
    double interval;
    int mvf;
    float mse_score = 0.0f;
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;

    // for MSE
    float m_val;
    int mif, mjf, mkf;
    int mir, mjr, mkr;
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    float diff;
    // ----------------------------------------------------------


    // --- INITIALIZE LOCAL VARIABLES ---------------------------
    ssd = &bst->ssd;
	
    if (parms->debug) {
	sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }
    // ----------------------------------------------------------

	
    // --- INITIALIZE GPU MEMORY --------------------------------
    bspline_cuda_h_push_coeff_lut(dev_ptrs, bxf);
    bspline_cuda_h_clear_score(dev_ptrs);
    bspline_cuda_h_clear_grad(dev_ptrs);
    // ----------------------------------------------------------


    plm_timer_start (&timer);	// <=== START TIMING HERE

    // generate histograms
    CUDA_bspline_MI_a_hist (dev_ptrs, mi_hist, fixed, moving, bxf);

    float tmp = 0;
    for (int zz=0; zz < mi_hist->fixed.bins; zz++) { tmp += f_hist[zz]; }
    printf ("f_hist total: %f\n", tmp);
    tmp = 0;
    for (int zz=0; zz < mi_hist->moving.bins; zz++) { tmp += m_hist[zz]; }
    printf ("m_hist total: %f\n", tmp);
    for (int zz=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) { tmp += j_hist[zz]; }
    printf ("j_hist total: %f\n", tmp);

	
    // Dump histogram images ??
    if (parms->xpm_hist_dump) {
	dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
	dump_hist (mi_hist, bst->it);
    }


    // Compute score
    num_vox = fixed->npix;
    num_vox_f = (float) num_vox;
    ssd->score = mi_hist_score (mi_hist, num_vox);

    // TEMP: Compute MSE
    printf ("MSE: ");
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

		// Get B-spline deformation vector
		pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
		qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

		// Compute coordinate of fixed image voxel
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		// Find correspondence in moving image
		mx = fx + dxyz[0];
		mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
		if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

		my = fy + dxyz[1];
		mj = (my - moving->offset[1]) / moving->pix_spacing[1];
		if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

		mz = fz + dxyz[2];
		mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
		if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

		// Compute linear interpolation fractions
		clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
		clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
		clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

		// Compute linearly interpolated moving image value
		mvf = (mkf * moving->dim[1] + mjf) * moving->dim[0] + mif;
		m_x1y1z1 = fx1 * fy1 * fz1;
		m_x2y1z1 = fx2 * fy1 * fz1;
		m_x1y2z1 = fx1 * fy2 * fz1;
		m_x2y2z1 = fx2 * fy2 * fz1;
		m_x1y1z2 = fx1 * fy1 * fz2;
		m_x2y1z2 = fx2 * fy1 * fz2;
		m_x1y2z2 = fx1 * fy2 * fz2;
		m_x2y2z2 = fx2 * fy2 * fz2;
		m_val = m_x1y1z1 * m_img[mvf]
		    + m_x2y1z1 * m_img[mvf+1]
		    + m_x1y2z1 * m_img[mvf+moving->dim[0]]
		    + m_x2y2z1 * m_img[mvf+moving->dim[0]+1]
		    + m_x1y1z2 * m_img[mvf+moving->dim[1]*moving->dim[0]] 
		    + m_x2y1z2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1]
		    + m_x1y2z2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]]
		    + m_x2y2z2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1];

		// compute (un-normalized) MSE
		diff = f_img[fv] - m_val;
		mse_score += diff * diff;
	    }
	}
    }

    printf ("%f\n", mse_score/fixed->npix);

    // TEMP: CPU Code
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));

    // PASS 2 - Compute gradient
    for (rk = 0, fk = bxf->roi_offset[2]; rk < bxf->roi_dim[2]; rk++, fk++) {
	p[2] = rk / bxf->vox_per_rgn[2];
	q[2] = rk % bxf->vox_per_rgn[2];
	fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
	for (rj = 0, fj = bxf->roi_offset[1]; rj < bxf->roi_dim[1]; rj++, fj++) {
	    p[1] = rj / bxf->vox_per_rgn[1];
	    q[1] = rj % bxf->vox_per_rgn[1];
	    fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
	    for (ri = 0, fi = bxf->roi_offset[0]; ri < bxf->roi_dim[0]; ri++, fi++) {
		long j_idxs[2];
		long m_idxs[2];
		long f_idxs[1];
		float fxs[2];
		float dS_dP;
		int debug;

		debug = 0;
		if (ri == 20 && rj == 20 && rk == 20) {
		    //debug = 1;
		}
		if (ri == 25 && rj == 25 && rk == 25) {
		    //debug = 1;
		}

		p[0] = ri / bxf->vox_per_rgn[0];
		q[0] = ri % bxf->vox_per_rgn[0];
		fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;

		// Get B-spline deformation vector
		pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
		qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

		// Compute coordinate of fixed image voxel
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		// Find correspondence in moving image
		mx = fx + dxyz[0];
		mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
		if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

		my = fy + dxyz[1];
		mj = (my - moving->offset[1]) / moving->pix_spacing[1];
		if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

		mz = fz + dxyz[2];
		mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
		if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

		dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

		// Compute quadratic interpolation fractions
		clamp_quadratic_interpolate_grad_inline (mi, moving->dim[0], miqs, fxqs);
		clamp_quadratic_interpolate_grad_inline (mj, moving->dim[1], mjqs, fyqs);
		clamp_quadratic_interpolate_grad_inline (mk, moving->dim[2], mkqs, fzqs);

		// PARTIAL VALUE INTERPOLATION - 6 neighborhood
		mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - fxqs[1] * dS_dP;
		dc_dv[1] -= - fyqs[1] * dS_dP;
		dc_dv[2] -= - fzqs[1] * dS_dP;

		mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[0];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - fxqs[0] * dS_dP;

		mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[2];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - fxqs[2] * dS_dP;

		mvf = (mkqs[1] * moving->dim[1] + mjqs[0]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[1] -= - fyqs[0] * dS_dP;

		mvf = (mkqs[1] * moving->dim[1] + mjqs[2]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[1] -= - fyqs[2] * dS_dP;

		mvf = (mkqs[0] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[2] -= - fzqs[0] * dS_dP;

		mvf = (mkqs[2] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[2] -= - fzqs[2] * dS_dP;

		dc_dv[0] = dc_dv[0] / moving->pix_spacing[0] / num_vox_f;
		dc_dv[1] = dc_dv[1] / moving->pix_spacing[1] / num_vox_f;
		dc_dv[2] = dc_dv[2] / moving->pix_spacing[2] / num_vox_f;

		if (parms->debug) {
		    //		    fprintf (fp, "%d %d %d %g %g %g\n", ri, rj, rk, dc_dv[0], dc_dv[1], dc_dv[2]);
		    fprintf (fp, "%d %d %d %g %g %g\n", 
			ri, rj, rk, 
			fxqs[0], fxqs[1], fxqs[2]);
		}

		bspline_update_grad_b_inline (bst, bxf, pidx, qidx, dc_dv);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    interval = plm_timer_report (&timer);

    report_score ("MI", bxf, bst, num_vox, interval);

    // calculate dC/dp and MI

    // calculate dp/dv

    // calculate dC/dv

    // calculate dv/dP

    // calculate dC/dP

	
    //	printf ("Time: %f\n", plm_timer_report (&timer));
	
}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_score_j_mse()
//
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_score_j_mse (
    BSPLINE_Parms* parms,
    Bspline_state *bst,
    BSPLINE_Xform* bxf,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Dev_Pointers_Bspline* dev_ptrs)
{
    // --- DECLARE LOCAL VARIABLES ------------------------------
    BSPLINE_Score* ssd;		// Holds the SSD "Score" information
    int num_vox;		// Holds # of voxels in the fixed volume
    float ssd_grad_norm;	// Holds the SSD Gradient's Norm
    float ssd_grad_mean;	// Holds the SSD Gradient's Mean
    Timer timer;

    static int it=0;	// Holds Iteration Number
    char debug_fn[1024];	// Debug message buffer
    FILE* fp = NULL;		// File Pointer to Debug File
    // ----------------------------------------------------------


    // --- INITIALIZE LOCAL VARIABLES ---------------------------
    ssd = &bst->ssd;
	
    if (parms->debug) {
	sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }
    // ----------------------------------------------------------


    plm_timer_start (&timer);	// <=== START TIMING HERE

	
    // --- INITIALIZE GPU MEMORY --------------------------------
    bspline_cuda_h_push_coeff_lut(dev_ptrs, bxf);
    bspline_cuda_h_clear_score(dev_ptrs);
    bspline_cuda_h_clear_grad(dev_ptrs);
    // ----------------------------------------------------------


	
    // --- LAUNCH STUB FUNCTIONS --------------------------------

    // Populate the score, dc_dv, and gradient
    bspline_cuda_j_stage_1(
	fixed,
	moving,
	moving_grad,
	bxf,
	parms,
	dev_ptrs);


    // Calculate the score and gradient
    // via sum reduction
    bspline_cuda_j_stage_2(
	parms,
	bxf,
	fixed,
	bxf->vox_per_rgn,
	fixed->dim,
	&(ssd->score),
	bst->ssd.grad, //ssd->grad,
	&ssd_grad_mean,
	&ssd_grad_norm,
	dev_ptrs,
	&num_vox);

    if (parms->debug) {
	fclose (fp);
    }

    // --- USER FEEDBACK ----------------------------------------
    report_score ("MSE", bxf, bst, num_vox, plm_timer_report (&timer));
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_cuda_score_i_mse()
//
////////////////////////////////////////////////////////////////////////////////
void bspline_cuda_score_i_mse (
    BSPLINE_Parms* parms,
    Bspline_state *bst,
    BSPLINE_Xform* bxf,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Dev_Pointers_Bspline* dev_ptrs)
{


    // --- DECLARE LOCAL VARIABLES ------------------------------
    BSPLINE_Score* ssd;	// Holds the SSD "Score" information
    int num_vox;		// Holds # of voxels in the fixed volume
    float ssd_grad_norm;	// Holds the SSD Gradient's Norm
    float ssd_grad_mean;	// Holds the SSD Gradient's Mean
    Timer timer;            // The timer

    static int it=0;	// Holds Iteration Number
    char debug_fn[1024];	// Debug message buffer
    FILE* fp = NULL;		// File Pointer to Debug File
    // ----------------------------------------------------------


    // --- INITIALIZE LOCAL VARIABLES ---------------------------
    ssd = &bst->ssd;
	
    if (parms->debug) {
	    sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	    fp = fopen (debug_fn, "w");
	}
    // ----------------------------------------------------------

    plm_timer_start (&timer);  // <=== START TIMING HERE
	
    // --- INITIALIZE GPU MEMORY --------------------------------
    bspline_cuda_h_push_coeff_lut(dev_ptrs, bxf);
    bspline_cuda_h_clear_score(dev_ptrs);
    bspline_cuda_h_clear_grad(dev_ptrs);
    // ----------------------------------------------------------


	
    // --- LAUNCH STUB FUNCTIONS --------------------------------

    // Populate the score, dc_dv, and gradient
    bspline_cuda_i_stage_1(
	fixed,
	moving,
	moving_grad,
	bxf,
	parms,
	dev_ptrs);


    // Calculate the score and gradient
    // via sum reduction
    bspline_cuda_j_stage_2(
	parms,
	bxf,
	fixed,
	bxf->vox_per_rgn,
	fixed->dim,
	&(ssd->score),
	bst->ssd.grad, //ssd->grad,
	&ssd_grad_mean,
	&ssd_grad_norm,
	dev_ptrs,
	&num_vox);

    if (parms->debug) {
	fclose (fp);
    }

    // --- USER FEEDBACK ----------------------------------------
    report_score ("MSE", bxf, bst, num_vox, plm_timer_report (&timer));
}
////////////////////////////////////////////////////////////////////////////////
