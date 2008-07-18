/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------
    REFS:
    http://en.wikipedia.org/wiki/B-spline
    http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
    http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    For multithreading
	On Win32: GetProcessAffinityMask, or GetSystemInfo
	    http://msdn.microsoft.com/en-us/library/ms810438.aspx
	Posix: 
	    http://ndevilla.free.fr/threads/index.html
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

extern void 
bspline_score_on_gpu_reference(BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
			       Volume *moving_grad);

#define round_int(x) ((x)>=0?(long)((x)+0.5):(long)(-(-(x)+0.5)))

void
bspline_default_parms (BSPLINE_Parms* parms)
{
    memset (parms, 0, sizeof(BSPLINE_Parms));
    parms->algorithm = BA_LBFGSB;
    parms->method = BM_CPU;
    parms->vox_per_rgn[0] = 30;
    parms->vox_per_rgn[1] = 30;
    parms->vox_per_rgn[2] = 30;
    parms->roi_dim[0] = 0;
    parms->roi_dim[1] = 0;
    parms->roi_dim[2] = 0;
    parms->roi_offset[0] = 0;
    parms->roi_offset[1] = 0;
    parms->roi_offset[2] = 0;
    parms->max_its = 10;
}

void
write_bspd (char* filename, BSPLINE_Parms* parms)
{
    BSPLINE_Data* bspd = &parms->bspd;
    FILE* fp;
	
    fp = fopen (filename, "wb");
    if (!fp) return;

    fprintf (fp, "MGH_GPUIT_BSP <experimental>\n");
    fprintf (fp, "vox_per_rgn = %d %d %d\n", parms->vox_per_rgn[0], parms->vox_per_rgn[1], parms->vox_per_rgn[2]);
    fprintf (fp, "roi_offset = %d %d %d\n", parms->roi_offset[0], parms->roi_offset[1], parms->roi_offset[2]);
    fprintf (fp, "roi_dim = %d %d %d\n", parms->roi_dim[0], parms->roi_dim[1], parms->roi_dim[2]);

    fprintf (fp, "rdims = %d %d %d\n", bspd->rdims[0], bspd->rdims[1], bspd->rdims[2]);
    fprintf (fp, "cdims = %d %d %d\n", bspd->cdims[0], bspd->cdims[1], bspd->cdims[2]);
    fprintf (fp, "num_coeff = %d\n", bspd->num_coeff);

#if defined (commentout)
    {
	/* This dumps in native, interleaved format */
	for (i = 0; i < bspd->num_coeff; i++) {
	    fprintf (fp, "%6.3f\n", bspd->coeff[i]);
	}
    }
#endif

    /* This dumps in itk-like planar format */
    {
	int i, j;
	for (i = 0; i < 3; i++) {
	    for (j = 0; j < bspd->num_coeff / 3; j++) {
		fprintf (fp, "%6.3f\n", bspd->coeff[j*3 + i]);
	    }
	}
    }		

    fclose (fp);
}

/* -----------------------------------------------------------------------
   Debugging routines
   ----------------------------------------------------------------------- */
static void
dump_parms (BSPLINE_Parms* parms)
{
    printf ("BSPLINE PARMS\n");
    printf ("max_its = %d\n", parms->max_its);
    printf ("vox_per_rgn = %d %d %d\n", parms->vox_per_rgn[0], parms->vox_per_rgn[1], parms->vox_per_rgn[2]);
    printf ("roi_offset = %d %d %d\n", parms->roi_offset[0], parms->roi_offset[1], parms->roi_offset[2]);
    printf ("roi_dim = %d %d %d\n", parms->roi_dim[0], parms->roi_dim[1], parms->roi_dim[2]);
}

void dump_gradient (BSPLINE_Data* bspd, BSPLINE_Score* ssd, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bspd->num_coeff; i++) {
	fprintf (fp, "%f\n", ssd->grad[i]);
    }
    fclose (fp);
}

void dump_coeff (BSPLINE_Data* bspd, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bspd->num_coeff; i++) {
	fprintf (fp, "%f\n", bspd->coeff[i]);
    }
    fclose (fp);
}

static void
dump_luts (BSPLINE_Parms* parms)
{
    int i, j, k, p;
    int tx, ty, tz;
    BSPLINE_Data* bspd = &parms->bspd;
    FILE* fp = fopen ("qlut.txt","wb");

    /* Dump q_lut */
    for (k = 0, p = 0; k < parms->vox_per_rgn[2]; k++) {
	for (j = 0; j < parms->vox_per_rgn[1]; j++) {
	    for (i = 0; i < parms->vox_per_rgn[0]; i++) {
		fprintf (fp, "%3d %3d %3d\n", k, j, i);
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
			    fprintf (fp, " %f", bspd->q_lut[p++]);
			}
		    }
		}
		fprintf (fp, "\n");
	    }
	}
    }
    fclose (fp);

    /* Test q_lut */
#if defined (commentout)
    printf ("Testing q_lut\n");
    for (j = 0; j < parms->vox_per_rgn[2] 
		 * parms->vox_per_rgn[1] 
		 * parms->vox_per_rgn[0]; j++) {
	float sum = 0.0;
	for (i = j*64; i < (j+1)*64; i++) {
	    sum += bspd->q_lut[i];
	}
	if (fabs(sum-1.0) > 1.e-7) {
	    printf ("%g ", fabs(sum-1.0));
	}
    }
    printf ("\n");
#endif

    fp = fopen ("clut.txt","wb");
    p = 0;
    for (k = 0; k < bspd->rdims[2]; k++) {
	for (j = 0; j < bspd->rdims[1]; j++) {
	    for (i = 0; i < bspd->rdims[0]; i++) {
		fprintf (fp, "%3d %3d %3d\n", k, j, i);
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
			    fprintf (fp, " %f", bspd->c_lut[p++]);
			}
		    }
		}
		fprintf (fp, "\n");
	    }
	}
    }
    fclose (fp);
}


/* -----------------------------------------------------------------------
   Reference code for alternate GPU-based data structure
   ----------------------------------------------------------------------- */
void
control_poimg_loop (BSPLINE_Data* bspd, Volume* fixed, BSPLINE_Parms* parms)
{
    int i, j, k;
    int rx, ry, rz;
    int vx, vy, vz;
    int cidx;
    float* img;

    img = (float*) fixed->img;

    /* Loop through cdim^3 control points */
    for (k = 0; k < bspd->cdims[2]; k++) {
	for (j = 0; j < bspd->cdims[1]; j++) {
	    for (i = 0; i < bspd->cdims[0]; i++) {

		/* Linear index of control point */
		cidx = k * bspd->cdims[1] * bspd->cdims[0]
		    + j * bspd->cdims[0] + i;

		/* Each control point has 64 regions */
		for (rz = 0; rz < 4; rz ++) {
		    for (ry = 0; ry < 4; ry ++) {
			for (rx = 0; rx < 4; rx ++) {

			    /* Some of the 64 regions are invalid. */
			    if (k + rz - 2 < 0) continue;
			    if (k + rz - 2 >= bspd->rdims[2]) continue;
			    if (j + ry - 2 < 0) continue;
			    if (j + ry - 2 >= bspd->rdims[1]) continue;
			    if (i + rx - 2 < 0) continue;
			    if (i + rx - 2 >= bspd->rdims[0]) continue;

			    /* Each region has vox_per_rgn^3 voxels */
			    for (vz = 0; vz < parms->vox_per_rgn[2]; vz ++) {
				for (vy = 0; vy < parms->vox_per_rgn[1]; vy ++) {
				    for (vx = 0; vx < parms->vox_per_rgn[0]; vx ++) {
					int img_idx[3], p;
					float img_val, coeff_val;

					/* Get (i,j,k) index of the voxel */
					img_idx[0] = parms->roi_offset[0] + (i + rx - 2) * parms->vox_per_rgn[0] + vx;
					img_idx[1] = parms->roi_offset[1] + (j + ry - 2) * parms->vox_per_rgn[1] + vy;
					img_idx[2] = parms->roi_offset[2] + (k + rz - 2) * parms->vox_per_rgn[2] + vz;

					/* Some of the pixels are invalid. */
					if (img_idx[0] > fixed->dim[0]) continue;
					if (img_idx[1] > fixed->dim[1]) continue;
					if (img_idx[2] > fixed->dim[2]) continue;

					/* Get the image value */
					p = img_idx[2] * fixed->dim[1] * fixed->dim[0] 
					    + img_idx[1] * fixed->dim[0] + img_idx[0];
					img_val = img[p];

					/* Get coefficient multiplier */
					p = vz * parms->vox_per_rgn[0] * parms->vox_per_rgn[1]
					    + vy * parms->vox_per_rgn[0] + vx;
					coeff_val = bspd->coeff[p];

					/* Here you would update the gradient: 
					    grad[cidx] += (fixed_val - moving_val) * coeff_val;
					*/
				    }
				}
			    }
			}
		    }
		}
	    }
	}
    }
}

void
bspline_display_coeff_stats (BSPLINE_Parms* parms)
{
    BSPLINE_Data* bspd = &parms->bspd;
    float cf_min, cf_avg, cf_max;
    int i;

    cf_avg = 0.0;
    cf_min = cf_max = bspd->coeff[0];
    for (i = 0; i < bspd->num_coeff; i++) {
	cf_avg += bspd->coeff[i];
	if (cf_min > bspd->coeff[i]) cf_min = bspd->coeff[i];
	if (cf_max < bspd->coeff[i]) cf_max = bspd->coeff[i];
    }
    printf ("CF (MIN,AVG,MAX) = %g %g %g\n", 
	    cf_min, cf_avg / bspd->num_coeff, cf_max);
}

void
bspline_set_coefficients (BSPLINE_Parms* parms, float val)
{
    BSPLINE_Data* bspd = &parms->bspd;
    int i;

    for (i = 0; i < bspd->num_coeff; i++) {
	bspd->coeff[i] = val;
    }
}

/* -----------------------------------------------------------------------
    qlut = Multiplier LUT
    clut = Index LUT
   ----------------------------------------------------------------------- */
void
bspline_initialize (BSPLINE_Parms* parms)
{
    BSPLINE_Data* bspd = &parms->bspd;
    int i, j, k, p;
    int tx, ty, tz;
    float *A, *B, *C;

    /* rdims is the number of regions */
    bspd->rdims[0] = 1 + (parms->roi_dim[0] - 1) / parms->vox_per_rgn[0];
    bspd->rdims[1] = 1 + (parms->roi_dim[1] - 1) / parms->vox_per_rgn[1];
    bspd->rdims[2] = 1 + (parms->roi_dim[2] - 1) / parms->vox_per_rgn[2];

    /* cdims is the number of control points */
    bspd->cdims[0] = 3 + bspd->rdims[0];
    bspd->cdims[1] = 3 + bspd->rdims[1];
    bspd->cdims[2] = 3 + bspd->rdims[2];

    /* total number of control points & coefficients */
    bspd->num_knots = bspd->cdims[0] * bspd->cdims[1] * bspd->cdims[2];
    bspd->num_coeff = bspd->cdims[0] * bspd->cdims[1] * bspd->cdims[2] * 3;

    /* Allocate coefficients */
    bspd->coeff = (float*) malloc (sizeof(float) * bspd->num_coeff);
    memset (bspd->coeff, 0, sizeof(float) * bspd->num_coeff);

    /* Allocate gradient */
    parms->ssd.grad = (float*) malloc (bspd->num_coeff * sizeof(float));

    /* Create q_lut */
    bspd->q_lut = (float*) malloc (sizeof(float) 
				 * parms->vox_per_rgn[0] 
				 * parms->vox_per_rgn[1] 
				 * parms->vox_per_rgn[2] 
				 * 64);
    A = (float*) malloc (sizeof(float) * parms->vox_per_rgn[0] * 4);
    B = (float*) malloc (sizeof(float) * parms->vox_per_rgn[1] * 4);
    C = (float*) malloc (sizeof(float) * parms->vox_per_rgn[2] * 4);

    for (i = 0; i < parms->vox_per_rgn[0]; i++) {
	float ii = ((float) i) / parms->vox_per_rgn[0];
	float t3 = ii*ii*ii;
	float t2 = ii*ii;
	float t1 = ii;
	A[i*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	A[i*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	A[i*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	A[i*4+3] = (1.0/6.0) * (+ 1.0 * t3);
    }
    for (j = 0; j < parms->vox_per_rgn[1]; j++) {
	float jj = ((float) j) / parms->vox_per_rgn[1];
	float t3 = jj*jj*jj;
	float t2 = jj*jj;
	float t1 = jj;
	B[j*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	B[j*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	B[j*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	B[j*4+3] = (1.0/6.0) * (+ 1.0 * t3);
    }
    for (k = 0; k < parms->vox_per_rgn[2]; k++) {
	float kk = ((float) k) / parms->vox_per_rgn[2];
	float t3 = kk*kk*kk;
	float t2 = kk*kk;
	float t1 = kk;
	C[k*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	C[k*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	C[k*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	C[k*4+3] = (1.0/6.0) * (+ 1.0 * t3);
    }

    p = 0;
    for (k = 0; k < parms->vox_per_rgn[2]; k++) {
	for (j = 0; j < parms->vox_per_rgn[1]; j++) {
	    for (i = 0; i < parms->vox_per_rgn[0]; i++) {
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
			    bspd->q_lut[p++] = A[i*4+tx] * B[j*4+ty] * C[k*4+tz];
			}
		    }
		}
	    }
	}
    }
    free (C);
    free (B);
    free (A);

    /* Create c_lut */
    bspd->c_lut = (int*) malloc (sizeof(int) 
				 * bspd->rdims[0] 
				 * bspd->rdims[1] 
				 * bspd->rdims[2] 
				 * 64);
    p = 0;
    for (k = 0; k < bspd->rdims[2]; k++) {
	for (j = 0; j < bspd->rdims[1]; j++) {
	    for (i = 0; i < bspd->rdims[0]; i++) {
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
			    bspd->c_lut[p++] = 
				    + (k + tz) * bspd->cdims[0] * bspd->cdims[1]
				    + (j + ty) * bspd->cdims[0] 
				    + (i + tx);
			}
		    }
		}
	    }
	}
    }

    //dump_luts (bspd);

    printf ("CDims = (%d %d %d)\n", bspd->cdims[0], bspd->cdims[1], 
	    bspd->cdims[2]);
}

void
bspline_free (BSPLINE_Parms* parms)
{
    BSPLINE_Data* bspd = &parms->bspd;
    free (bspd->coeff);
    free (bspd->q_lut);
    free (bspd->c_lut);
    free (parms->ssd.grad);
}

inline void
bspline_interp_pix (float out[3], BSPLINE_Data* bspd, int p[3], int qidx)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bspd->q_lut[qidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = (p[2] + k) * bspd->cdims[1] * bspd->cdims[0]
			+ (p[1] + j) * bspd->cdims[0]
			+ (p[0] + i);
		cidx = cidx * 3;
		out[0] += q_lut[m] * bspd->coeff[cidx+0];
		out[1] += q_lut[m] * bspd->coeff[cidx+1];
		out[2] += q_lut[m] * bspd->coeff[cidx+2];
		m ++;
	    }
	}
    }
}

inline void
bspline_interp_pix_b_inline (float out[3], BSPLINE_Data* bspd, int pidx, int qidx)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bspd->q_lut[qidx*64];
    int* c_lut = &bspd->c_lut[pidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = 3 * c_lut[m];
		out[0] += q_lut[m] * bspd->coeff[cidx+0];
		out[1] += q_lut[m] * bspd->coeff[cidx+1];
		out[2] += q_lut[m] * bspd->coeff[cidx+2];
		m ++;
	    }
	}
    }
}

void
bspline_interp_pix_b (float out[3], BSPLINE_Data* bspd, int pidx, int qidx)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bspd->q_lut[qidx*64];
    int* c_lut = &bspd->c_lut[pidx*64];

    out[0] = out[1] = out[2] = 0;
    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = 3 * c_lut[m];
		out[0] += q_lut[m] * bspd->coeff[cidx+0];
		out[1] += q_lut[m] * bspd->coeff[cidx+1];
		out[2] += q_lut[m] * bspd->coeff[cidx+2];
		m ++;
	    }
	}
    }
}

void
bspline_interpolate_vf (Volume* interp, 
			BSPLINE_Parms* parms)
{
    BSPLINE_Data* bspd = &parms->bspd;
    int i, j, k, v;
    int p[3];
    int q[3];
    float* out;
    float* img = (float*) interp->img;
    int qidx;

    memset (img, 0, interp->npix*3*sizeof(float));
    for (k = 0; k < parms->roi_dim[2]; k++) {
	p[2] = k / parms->vox_per_rgn[2];
	q[2] = k % parms->vox_per_rgn[2];
	for (j = 0; j < parms->roi_dim[1]; j++) {
	    p[1] = j / parms->vox_per_rgn[1];
	    q[1] = j % parms->vox_per_rgn[1];
	    for (i = 0; i < parms->roi_dim[0]; i++) {
		p[0] = i / parms->vox_per_rgn[0];
		q[0] = i % parms->vox_per_rgn[0];
		qidx = q[2] * parms->vox_per_rgn[1] * parms->vox_per_rgn[0]
			+ q[1] * parms->vox_per_rgn[0] + q[0];
		v = (k+parms->roi_offset[2]) * interp->dim[0] * interp->dim[1]
			+ (j+parms->roi_offset[1]) * interp->dim[0] + (i+parms->roi_offset[0]);
		out = &img[3*v];
		bspline_interp_pix (out, bspd, p, qidx);
	    }
	}
    }
}

inline void
bspline_update_grad (BSPLINE_Parms* parms, int p[3], int qidx, float dc_dv[3])
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
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

inline void
bspline_update_grad_b_inline (BSPLINE_Parms* parms,  
		     int pidx, int qidx, float dc_dv[3])
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    int i, j, k, m;
    int cidx;
    float* q_lut = &bspd->q_lut[qidx*64];
    int* c_lut = &bspd->c_lut[pidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = 3 * c_lut[m];
		ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
		ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
		ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
		m ++;
	    }
	}
    }
}

void
bspline_update_grad_b (BSPLINE_Parms* parms,  
		     int pidx, int qidx, float dc_dv[3])
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    int i, j, k, m;
    int cidx;
    float* q_lut = &bspd->q_lut[qidx*64];
    int* c_lut = &bspd->c_lut[pidx*64];

    m = 0;
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		cidx = 3 * c_lut[m];
		ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
		ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
		ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
		m ++;
	    }
	}
    }
}

/* Clipping is done using clamping.  You should have "air" as the outside
   voxel so pixels can be clamped to air.  */
inline void
clip_and_interpolate_obsolete (
    Volume* moving,	/* Moving image */
    float* dxyz,	/* Vector displacement of current voxel */
    float* dxyzf,	/* Floor of vector displacement */
    int d,		/* 0 for x, 1 for y, 2 for z */
    int* maf,		/* x, y, or z coord of "floor" pixel in moving img */
    int* mar,		/* x, y, or z coord of "round" pixel in moving img */
    int a,		/* Index of base voxel (before adding displacemnt) */
    float* fa1,		/* Fraction of interpolant for lower index voxel */
    float* fa2		/* Fraction of interpolant for upper index voxel */
)
{
    dxyzf[d] = floor (dxyz[d]);
    *maf = a + (int) dxyzf[d];
    *mar = a + round_int (dxyz[d]);
    *fa2 = dxyz[d] - dxyzf[d];
    if (*maf < 0) {
	*maf = 0;
	*mar = 0;
	*fa2 = 0.0f;
    } else if (*maf >= moving->dim[d] - 1) {
	*maf = moving->dim[d] - 2;
	*mar = moving->dim[d] - 1;
	*fa2 = 1.0f;
    }
    *fa1 = 1.0f - *fa2;
}

/* Clipping is done using clamping.  You should have "air" as the outside
   voxel so pixels can be clamped to air.  */
inline void
clamp_and_interpolate_inline (
    float ma,           /* (Unrounded) pixel coordinate (in vox) */
    int dmax,		/* Maximum coordinate in this dimension */
    int* maf,		/* x, y, or z coord of "floor" pixel in moving img */
    int* mar,		/* x, y, or z coord of "round" pixel in moving img */
    float* fa1,		/* Fraction of interpolant for lower index voxel */
    float* fa2		/* Fraction of interpolant for upper index voxel */
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

void
clamp_and_interpolate (
    float ma,           /* (Unrounded) pixel coordinate (in vox) */
    int dmax,		/* Maximum coordinate in this dimension */
    int* maf,		/* x, y, or z coord of "floor" pixel in moving img */
    int* mar,		/* x, y, or z coord of "round" pixel in moving img */
    float* fa1,		/* Fraction of interpolant for lower index voxel */
    float* fa2		/* Fraction of interpolant for upper index voxel */
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

/* This is slower than version B, but yields a smoother cost function 
   for use by L-BFGS-B.  It uses linear interpolation of moving image, 
   and nearest neighbor interpolation of gradient */
void
bspline_score_c (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		 Volume *moving_grad)
{
    BSPLINE_Data* bspd = &parms->bspd;
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

    start_clock = clock();

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
		bspline_interp_pix_b_inline (dxyz, bspd, pidx, qidx);

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = (mx - moving->offset[0]) / moving->pix_spacing[0];
		if (mi < -0.5 || mi > moving->dim[0] - 0.5) continue;

		my = fy + dxyz[1];
		mj = (my - moving->offset[1]) / moving->pix_spacing[1];
		if (mj < -0.5 || mj > moving->dim[1] - 0.5) continue;

		mz = fz + dxyz[2];
		mk = (mz - moving->offset[2]) / moving->pix_spacing[2];
		if (mk < -0.5 || mk > moving->dim[2] - 0.5) continue;

		/* Compute interpolation fractions */
		clamp_and_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
		clamp_and_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
		clamp_and_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

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
		bspline_update_grad_b_inline (parms, pidx, qidx, dc_dv);
		
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
    printf ("GET VALUE+DERIVATIVE: %6.3f [%6d] %6.3f %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

/* This is the fastest known version.  It does nearest neighbors 
   interpolation of both moving image and gradient which doesn't 
   work with stock L-BFGS-B optimizer. */
void
bspline_score_b (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		 Volume *moving_grad)
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    int i;
    int ri, rj, rk;
    int fi, fj, fk, fv;
    int mi, mj, mk, mv;
    float fx, fy, fz;
    float mx, my, mz;
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;

    start_clock = clock();

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
		qidx = q[2] * parms->vox_per_rgn[1] * parms->vox_per_rgn[0]
			+ q[1] * parms->vox_per_rgn[0] + q[0];
		bspline_interp_pix_b (dxyz, bspd, pidx, qidx);

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = round_int((mx - moving->offset[0]) / moving->pix_spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = round_int((my - moving->offset[1]) / moving->pix_spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = round_int((mz - moving->offset[2]) / moving->pix_spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;
		mv = (mk * moving->dim[1] + mj) * moving->dim[0] + mi;

		/* Compute intensity difference */
		diff = f_img[fv] - m_img[mv];

		/* Compute spatial gradient using nearest neighbors */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */

		bspline_update_grad_b (parms, pidx, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bspd->num_coeff; i++) {
	ssd->grad[i] = ssd->grad[i] / num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bspd->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }

    end_clock = clock();
    printf ("Single iteration CPU [b] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
    printf ("NUM_VOX = %d\n", num_vox);
    printf ("MSE = %g\n", ssd->score);
    printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    printf ("GRAD_NORM = %g\n", ssd_grad_norm);
}

void
bspline_score_a (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		 Volume *moving_grad)
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    int i;
    int ri, rj, rk;
    int fi, fj, fk, fv;
    int mi, mj, mk, mv;
    float fx, fy, fz;
    float mx, my, mz;
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int qidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;

    start_clock = clock();

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
		qidx = q[2] * parms->vox_per_rgn[1] * parms->vox_per_rgn[0]
			+ q[1] * parms->vox_per_rgn[0] + q[0];
		bspline_interp_pix (dxyz, bspd, p, qidx);

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = round_int((mx - moving->offset[0]) / moving->pix_spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = round_int((my - moving->offset[1]) / moving->pix_spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = round_int((mz - moving->offset[2]) / moving->pix_spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;
		mv = (mk * moving->dim[1] + mj) * moving->dim[0] + mi;

		/* Compute intensity difference */
		diff = f_img[fv] - m_img[mv];

		/* Compute spatial gradient using nearest neighbors */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */
		bspline_update_grad (parms, p, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bspd->num_coeff; i++) {
	ssd->grad[i] /= num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bspd->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }
    end_clock = clock();
    printf ("Single iteration CPU [a] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);

    printf ("MSE = %g\n", ssd->score);
    printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    printf ("GRAD_NORM = %g\n", ssd_grad_norm);
}

void
bspline_score (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
	       Volume *moving_grad)
{
#if HAVE_BROOK
#if BUILD_BSPLINE_BROOK
    if (parms->method == BM_BROOK) {
	printf("Using GPU. \n");
	bspline_score_on_gpu_reference (parms, fixed, moving, moving_grad);
	return;
    }
#endif
#endif
    printf("Using CPU. \n");
    bspline_score_c (parms, fixed, moving, moving_grad);
//    bspline_score_b (parms, fixed, moving, moving_grad);
//    bspline_score_a (parms, fixed, moving, moving_grad);
}

void
bspline_optimize_steepest (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		  Volume *moving_grad)
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    int i, it;
    float a = 0.003f;
    float alpha = 0.5f, A = 10.0f;

    bspline_set_coefficients (parms, 0.0);

    /* Get score and gradient */
    bspline_score (parms, fixed, moving, moving_grad);
    /* Give a little feedback to the user */
    bspline_display_coeff_stats (parms);

    for (it = 0; it < parms->max_its; it++) {
	float gamma;

	printf ("Beginning iteration %d\n", it);

	/* Update coefficients */
	gamma = a / pow(it + A, alpha);
	for (i = 0; i < bspd->num_coeff; i++) {
	    bspd->coeff[i] = bspd->coeff[i] + gamma * ssd->grad[i];
	}

	/* Get score and gradient */
	bspline_score (parms, fixed, moving, moving_grad);

	/* Give a little feedback to the user */
	bspline_display_coeff_stats (parms);
    }
}

void
bspline_optimize_debug (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		  Volume *moving_grad, char* method)
{
    BSPLINE_Data* bspd = &parms->bspd;
    BSPLINE_Score* ssd = &parms->ssd;
    int i, it;
    float step = 0.001f;

    bspline_set_coefficients (parms, 0.0);

    /* Get score and gradient */
    bspline_score (parms, fixed, moving, moving_grad);
    //dump_gradient (bspd, &ssd_fixed, "grad.txt");

    /* Give a little feedback to the user */
    bspline_display_coeff_stats (parms);

    for (it = 0; it < parms->max_its; it++) {
	printf ("Beginning iteration %d\n", it);

	/* Update coefficients */
	for (i = 0; i < bspd->num_coeff; i++) {
	    bspd->coeff[i] = bspd->coeff[i] + step * ssd->grad[i];
	}
	//dump_coeff (bspd, "coeff.txt");

	/* Get score and gradient */
	bspline_score (parms, fixed, moving, moving_grad);

	/* Give a little feedback to the user */
	bspline_display_coeff_stats (parms);
	//exit (0);
    }
}

void
bspline_optimize (BSPLINE_Parms *parms, Volume *fixed, Volume *moving, 
		  Volume *moving_grad)
{
    dump_parms (parms);
    if (parms->algorithm == BA_LBFGSB) {
#if defined (HAVE_F2C_LIBRARY)
	bspline_optimize_lbfgsb (parms, fixed, moving, moving_grad);
#else
	fprintf (stderr, 
	    "Sorry, LBFGSB not compiled for this platform (f2c library missing).\n"
	    "Reverting to steepest descent.\n"
	    );
	bspline_optimize_steepest (parms, fixed, moving, moving_grad);
#endif
    } else {
	bspline_optimize_steepest (parms, fixed, moving, moving_grad);
#if defined (commentout)
	bspline_optimize_debug (parms, fixed, moving, moving_grad);
#endif
    }
}
