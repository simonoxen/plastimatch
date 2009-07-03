/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -------------------------------------------------------------------------

    B-Spline basics:
	http://en.wikipedia.org/wiki/B-spline
	http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
	http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    For multithreading - how to get number of processors?
	On Win32: GetProcessAffinityMask, or GetSystemInfo
	    http://msdn.microsoft.com/en-us/library/ms810438.aspx
	Posix: 
	    http://ndevilla.free.fr/threads/index.html

    Proposed variable naming:
	Fixed image voxel                   (f[3]), fidx <currently (fi,fj,fk),fv>
	Moving image voxel                  (m[3]), midx < ditto >
	    - what about ROI's              ?
	    - what about physical coords    ?
	Tile (fixed)                        (t[3]), tidx <currently p[3]>
	Offset within tile (fixed)          (o[3]), oidx <currently q[3]>
	Control point                       (c[3]), cidx <currently (i,j,k), m>
	Coefficient array                   ?                <currently cidx>
	Multiplier LUT                      qidx
	Index LUT                           pidx

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
#include "logfile.h"
#if (HAVE_CUDA)
#include "bspline_cuda.h"
#endif
#include "mathutil.h"


gpuit_EXPORT
void
bspline_parms_set_default (BSPLINE_Parms* parms)
{
    memset (parms, 0, sizeof(BSPLINE_Parms));
    parms->threading = BTHR_CPU;
    parms->optimization = BOPT_LBFGSB;
    parms->metric = BMET_MSE;
    parms->implementation = '\0';
    parms->max_its = 10;
    parms->convergence_tol = 0.1;
    parms->convergence_tol_its = 4;
    parms->debug = 0;

    parms->mi_hist.f_hist = 0;
    parms->mi_hist.m_hist = 0;
    parms->mi_hist.j_hist = 0;
    parms->mi_hist.fixed.bins = 20;
    parms->mi_hist.moving.bins = 20;
}

gpuit_EXPORT
void
bspline_xform_set_default (BSPLINE_Xform* bxf)
{
    int d;

    memset (bxf, 0, sizeof(BSPLINE_Xform));

    for (d = 0; d < 3; d++) {
	bxf->img_origin[d] = 0.0f;
	bxf->img_spacing[d] = 1.0f;
	bxf->img_dim[d] = 0;
	bxf->roi_offset[d] = 0;
	bxf->roi_dim[d] = 0;
	bxf->vox_per_rgn[d] = 30;
	bxf->grid_spac[d] = 30.0f;
    }
}

gpuit_EXPORT
void
write_bxf (char* filename, BSPLINE_Xform* bxf)
{
    FILE* fp;
	
    fp = fopen (filename, "wb");
    if (!fp) return;

    fprintf (fp, "MGH_GPUIT_BSP <experimental>\n");
    fprintf (fp, "img_origin = %f %f %f\n", bxf->img_origin[0], bxf->img_origin[1], bxf->img_origin[2]);
    fprintf (fp, "img_spacing = %f %f %f\n", bxf->img_spacing[0], bxf->img_spacing[1], bxf->img_spacing[2]);
    fprintf (fp, "img_dim = %d %d %d\n", bxf->img_dim[0], bxf->img_dim[1], bxf->img_dim[2]);
    fprintf (fp, "roi_offset = %d %d %d\n", bxf->roi_offset[0], bxf->roi_offset[1], bxf->roi_offset[2]);
    fprintf (fp, "roi_dim = %d %d %d\n", bxf->roi_dim[0], bxf->roi_dim[1], bxf->roi_dim[2]);
    fprintf (fp, "vox_per_rgn = %d %d %d\n", bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    /* No need to save grid_spac */

#if defined (commentout)
    {
	/* This dumps in native, interleaved format */
	for (i = 0; i < bxf->num_coeff; i++) {
	    fprintf (fp, "%6.3f\n", bxf->coeff[i]);
	}
    }
#endif

    /* This dumps in itk-like planar format */
    {
	int i, j;
	for (i = 0; i < 3; i++) {
	    for (j = 0; j < bxf->num_coeff / 3; j++) {
		//fprintf (fp, "%6.3f\n", bxf->coeff[j*3 + i]);
		fprintf (fp, "%f\n", bxf->coeff[j*3 + i]);
	    }
	}
    }		

    fclose (fp);
}

gpuit_EXPORT
BSPLINE_Xform* 
read_bxf (char* filename)
{
    BSPLINE_Xform* bxf;
    char buf[1024];
    FILE* fp;
    int rc;
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    int roi_offset[3];		 /* Position of first vox in ROI (in vox) */
    int roi_dim[3];		 /* Dimension of ROI (in vox) */
    int vox_per_rgn[3];		 /* Knot spacing (in vox) */

    fp = fopen (filename, "r");
    if (!fp) return 0;

    /* Initialize parms */
    bxf = (BSPLINE_Xform*) malloc (sizeof(BSPLINE_Xform));
    bspline_xform_set_default (bxf);

    /* Skip first line */
    fgets (buf, 1024, fp);

    /* Read header */
    rc = fscanf (fp, "img_origin = %f %f %f\n", &img_origin[0], &img_origin[1], &img_origin[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_origin): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "img_spacing = %f %f %f\n", &img_spacing[0], &img_spacing[1], &img_spacing[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_spacing): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "img_dim = %d %d %d\n", &img_dim[0], &img_dim[1], &img_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_dim): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "roi_offset = %d %d %d\n", &roi_offset[0], &roi_offset[1], &roi_offset[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (roi_offset): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "roi_dim = %d %d %d\n", &roi_dim[0], &roi_dim[1], &roi_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (roi_dim): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "vox_per_rgn = %d %d %d\n", &vox_per_rgn[0], &vox_per_rgn[1], &vox_per_rgn[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (vox_per_rgn): %s\n", filename);
	goto free_exit;
    }

    /* Allocate memory and build LUTs */
    bspline_xform_initialize (bxf, img_origin, img_spacing, img_dim,
		roi_offset, roi_dim, vox_per_rgn);

    /* This loads from itk-like planar format */
    {
	int i, j;
	for (i = 0; i < 3; i++) {
	    for (j = 0; j < bxf->num_coeff / 3; j++) {
		rc = fscanf (fp, "%f\n", &bxf->coeff[j*3 + i]);
		if (rc != 1) {
		    logfile_printf ("Error parsing input xform (idx = %d,%d): %s\n", i, j, filename);
		    bspline_xform_free (bxf);
		    goto free_exit;
		}
	    }
	}
    }

    fclose (fp);
    return bxf;

free_exit:
    free (bxf);
    return 0;
}

/* -----------------------------------------------------------------------
   Debugging routines
   ----------------------------------------------------------------------- */
static void
log_parms (BSPLINE_Parms* parms)
{
    logfile_printf ("BSPLINE PARMS\n");
    logfile_printf ("max_its = %d\n", parms->max_its);
}

static void
log_bxf_header (BSPLINE_Xform* bxf)
{
    logfile_printf ("BSPLINE XFORM HEADER\n");
    logfile_printf ("vox_per_rgn = %d %d %d\n", bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("roi_offset = %d %d %d\n", bxf->roi_offset[0], bxf->roi_offset[1], bxf->roi_offset[2]);
    logfile_printf ("roi_dim = %d %d %d\n", bxf->roi_dim[0], bxf->roi_dim[1], bxf->roi_dim[2]);
}

void
dump_gradient (BSPLINE_Xform* bxf, BSPLINE_Score* ssd, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bxf->num_coeff; i++) {
	fprintf (fp, "%f\n", ssd->grad[i]);
    }
    fclose (fp);
}

void
dump_coeff (BSPLINE_Xform* bxf, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bxf->num_coeff; i++) {
	fprintf (fp, "%f\n", bxf->coeff[i]);
    }
    fclose (fp);
}

void
dump_luts (BSPLINE_Xform* bxf)
{
    int i, j, k, p;
    int tx, ty, tz;
    FILE* fp = fopen ("qlut.txt","wb");

    /* Dump q_lut */
    for (k = 0, p = 0; k < bxf->vox_per_rgn[2]; k++) {
	for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
	    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
		fprintf (fp, "%3d %3d %3d\n", k, j, i);
		for (tz = 0; tz < 4; tz++) {
		    for (ty = 0; ty < 4; ty++) {
			for (tx = 0; tx < 4; tx++) {
			    fprintf (fp, " %f", bxf->q_lut[p++]);
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
    for (j = 0; j < bxf->vox_per_rgn[2] 
		 * bxf->vox_per_rgn[1] 
		 * bxf->vox_per_rgn[0]; j++) {
	float sum = 0.0;
	for (i = j*64; i < (j+1)*64; i++) {
	    sum += bxf->q_lut[i];
	}
	if (fabs(sum-1.0) > 1.e-7) {
	    printf ("%g ", fabs(sum-1.0));
	}
    }
    printf ("\n");
#endif

    fp = fopen ("clut.txt","wb");
    p = 0;
    for (k = 0; k < bxf->rdims[2]; k++) {
		for (j = 0; j < bxf->rdims[1]; j++) {
			for (i = 0; i < bxf->rdims[0]; i++) {
				
				fprintf (fp, "%3d %3d %3d\n", k, j, i);
		
				for (tz = 0; tz < 4; tz++) {
					for (ty = 0; ty < 4; ty++) {
						for (tx = 0; tx < 4; tx++) {
							fprintf (fp, " %d", bxf->c_lut[p++]);
						}
					}
				}

				fprintf (fp, "\n");
			}
		}
    }
    fclose (fp);
}

void
dump_hist (BSPLINE_MI_Hist* mi_hist, char* fn)
{
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
    int i, j, v;
    FILE* fp;

    fp = fopen (fn, "wb");
    if (!fp) return;

    fprintf (fp, "Fixed hist\n");
    for (i = 0; i < mi_hist->fixed.bins; i++) {
	fprintf (fp, "[%2d] %20.3g\n", i, f_hist[i]);
    }
    fprintf (fp, "Moving hist\n");
    for (i = 0; i < mi_hist->moving.bins; i++) {
	fprintf (fp, "[%2d] %20.3g\n", i, m_hist[i]);
    }
    fprintf (fp, "Joint hist\n");
    for (i = 0, v = 0; i < mi_hist->fixed.bins; i++) {
	for (j = 0; j < mi_hist->moving.bins; j++, v++) {
	    if (j_hist[v] > 0) {
		fprintf (fp, "[%2d, %2d, %3d] %20.3g\n", i, j, v, j_hist[v]);
	    }
	}
    }
    fclose (fp);
}

/* -----------------------------------------------------------------------
   Reference code for alternate GPU-based data structure
   ----------------------------------------------------------------------- */
void
control_poimg_loop (BSPLINE_Xform* bxf, Volume* fixed)
{
    int i, j, k;
    int rx, ry, rz;
    int vx, vy, vz;
    int cidx;
    float* img;

    img = (float*) fixed->img;

    /* Loop through cdim^3 control points */
    for (k = 0; k < bxf->cdims[2]; k++) {
	for (j = 0; j < bxf->cdims[1]; j++) {
	    for (i = 0; i < bxf->cdims[0]; i++) {

		/* Linear index of control point */
		cidx = k * bxf->cdims[1] * bxf->cdims[0]
		    + j * bxf->cdims[0] + i;

		/* Each control point has 64 regions */
		for (rz = 0; rz < 4; rz ++) {
		    for (ry = 0; ry < 4; ry ++) {
			for (rx = 0; rx < 4; rx ++) {

			    /* Some of the 64 regions are invalid. */
			    if (k + rz - 2 < 0) continue;
			    if (k + rz - 2 >= bxf->rdims[2]) continue;
			    if (j + ry - 2 < 0) continue;
			    if (j + ry - 2 >= bxf->rdims[1]) continue;
			    if (i + rx - 2 < 0) continue;
			    if (i + rx - 2 >= bxf->rdims[0]) continue;

			    /* Each region has vox_per_rgn^3 voxels */
			    for (vz = 0; vz < bxf->vox_per_rgn[2]; vz ++) {
				for (vy = 0; vy < bxf->vox_per_rgn[1]; vy ++) {
				    for (vx = 0; vx < bxf->vox_per_rgn[0]; vx ++) {
					int img_idx[3], p;
					float img_val, coeff_val;

					/* Get (i,j,k) index of the voxel */
					img_idx[0] = bxf->roi_offset[0] + (i + rx - 2) * bxf->vox_per_rgn[0] + vx;
					img_idx[1] = bxf->roi_offset[1] + (j + ry - 2) * bxf->vox_per_rgn[1] + vy;
					img_idx[2] = bxf->roi_offset[2] + (k + rz - 2) * bxf->vox_per_rgn[2] + vz;

					/* Some of the pixels are invalid. */
					if (img_idx[0] > fixed->dim[0]) continue;
					if (img_idx[1] > fixed->dim[1]) continue;
					if (img_idx[2] > fixed->dim[2]) continue;

					/* Get the image value */
					p = img_idx[2] * fixed->dim[1] * fixed->dim[0] 
					    + img_idx[1] * fixed->dim[0] + img_idx[0];
					img_val = img[p];

					/* Get coefficient multiplier */
					p = vz * bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1]
					    + vy * bxf->vox_per_rgn[0] + vx;
					coeff_val = bxf->coeff[p];

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
bspline_display_coeff_stats (BSPLINE_Xform* bxf)
{
    float cf_min, cf_avg, cf_max;
    int i;

    cf_avg = 0.0;
    cf_min = cf_max = bxf->coeff[0];
    for (i = 0; i < bxf->num_coeff; i++) {
	cf_avg += bxf->coeff[i];
	if (cf_min > bxf->coeff[i]) cf_min = bxf->coeff[i];
	if (cf_max < bxf->coeff[i]) cf_max = bxf->coeff[i];
    }
    logfile_printf ("CF (MIN,AVG,MAX) = %g %g %g\n", 
	    cf_min, cf_avg / bxf->num_coeff, cf_max);
}

void
bspline_set_coefficients (BSPLINE_Xform* bxf, float val)
{
    int i;

    for (i = 0; i < bxf->num_coeff; i++) {
	bxf->coeff[i] = val;
    }
}

/* -----------------------------------------------------------------------
    qlut = Multiplier LUT
    clut = Index LUT

    Inputs: roi_dim, vox_per_rgn.
   ----------------------------------------------------------------------- */
gpuit_EXPORT
void
bspline_xform_initialize (
	BSPLINE_Xform* bxf,	     /* Output: bxf is initialized */
	float img_origin[3],         /* Image origin (in mm) */
	float img_spacing[3],        /* Image spacing (in mm) */
	int img_dim[3],              /* Image size (in vox) */
	int roi_offset[3],	     /* Position of first vox in ROI (in vox) */
	int roi_dim[3],		     /* Dimension of ROI (in vox) */
	int vox_per_rgn[3])	     /* Knot spacing (in vox) */
{
    int d;
    int i, j, k, p;
    int tx, ty, tz;
    float *A, *B, *C;

    logfile_printf ("bspline_xform_initialize\n");
    for (d = 0; d < 3; d++) {
		/* copy input parameters over */
		bxf->img_origin[d] = img_origin[d];
		bxf->img_spacing[d] = img_spacing[d];
		bxf->img_dim[d] = img_dim[d];
		bxf->roi_offset[d] = roi_offset[d];
		bxf->roi_dim[d] = roi_dim[d];
		bxf->vox_per_rgn[d] = vox_per_rgn[d];

		/* grid spacing is in mm */
		bxf->grid_spac[d] = bxf->vox_per_rgn[d] * bxf->img_spacing[d];

		/* rdims is the number of regions */
		bxf->rdims[d] = 1 + (bxf->roi_dim[d] - 1) / bxf->vox_per_rgn[d];

		/* cdims is the number of control points */
		bxf->cdims[d] = 3 + bxf->rdims[d];
    }

    /* total number of control points & coefficients */
    bxf->num_knots = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2];
    bxf->num_coeff = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2] * 3;

    /* Allocate coefficients */
    bxf->coeff = (float*) malloc (sizeof(float) * bxf->num_coeff);
    memset (bxf->coeff, 0, sizeof(float) * bxf->num_coeff);

    /* Create q_lut */
    bxf->q_lut = (float*) malloc (sizeof(float) 
				 * bxf->vox_per_rgn[0] 
				 * bxf->vox_per_rgn[1] 
				 * bxf->vox_per_rgn[2] 
				 * 64);
    A = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[0] * 4);
    B = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[1] * 4);
    C = (float*) malloc (sizeof(float) * bxf->vox_per_rgn[2] * 4);
 
    for (i = 0; i < bxf->vox_per_rgn[0]; i++) {
		float ii = ((float) i) / bxf->vox_per_rgn[0];
		float t3 = ii*ii*ii;
		float t2 = ii*ii;
		float t1 = ii;
		A[i*4+0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
		A[i*4+1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
		A[i*4+2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
		A[i*4+3] = (1.0/6.0) * (+ 1.0 * t3);
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
    }

    p = 0;
    for (k = 0; k < bxf->vox_per_rgn[2]; k++) {
		for (j = 0; j < bxf->vox_per_rgn[1]; j++) {
			for (i = 0; i < bxf->vox_per_rgn[0]; i++) {

				for (tz = 0; tz < 4; tz++) {
					for (ty = 0; ty < 4; ty++) {
						for (tx = 0; tx < 4; tx++) {
							bxf->q_lut[p++] = A[i*4+tx] * B[j*4+ty] * C[k*4+tz];
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
    bxf->c_lut = (int*) malloc (sizeof(int) 
				 * bxf->rdims[0] 
				 * bxf->rdims[1] 
				 * bxf->rdims[2] 
				 * 64);
    p = 0;
    for (k = 0; k < bxf->rdims[2]; k++) {
		for (j = 0; j < bxf->rdims[1]; j++) {
			for (i = 0; i < bxf->rdims[0]; i++) {
				for (tz = 0; tz < 4; tz++) {
					for (ty = 0; ty < 4; ty++) {
						for (tx = 0; tx < 4; tx++) {
							bxf->c_lut[p++] = 
								+ (k + tz) * bxf->cdims[0] * bxf->cdims[1]
								+ (j + ty) * bxf->cdims[0] 
								+ (i + tx);
						}
					}
				}
			}
		}
    }

    //dump_luts (bxf);

    logfile_printf ("rdims = (%d,%d,%d)\n", bxf->rdims[0], bxf->rdims[1], bxf->rdims[2]);
    logfile_printf ("vox_per_rgn = (%d,%d,%d)\n", bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("cdims = (%d %d %d)\n", bxf->cdims[0], bxf->cdims[1], bxf->cdims[2]);
}

/* This extends the bspline grid.  Note, that the new roi_offset 
    in the bxf will not be the same as the one requested, because 
    bxf routines implicitly require that the first voxel of the 
    ROI matches the position of the control point. */
/* GCS -- Is there an implicit assumption that the roi_origin > 0? */
void
bspline_xform_extend (
	BSPLINE_Xform* bxf,	     /* Output: bxf is initialized */
	int new_roi_offset[3],	     /* Position of first vox in ROI (in vox) */
	int new_roi_dim[3])	     /* Dimension of ROI (in vox) */
{
    int d;
    int roi_offset_diff[3];
    int roi_corner_diff[3];
    int eb[3], ea[3];  /* # of control points to "extend before" and "extend after" existing grid */
    int extend_needed = 0;
    int new_rdims[3];
    int new_cdims[3];
    int new_num_knots;
    int new_num_coeff;
    float* new_coeff;
    int old_idx;
    int i, j, k;

    for (d = 0; d < 3; d++) {
	roi_offset_diff[d] = new_roi_offset[d] - bxf->roi_offset[d];
	roi_corner_diff[d] = (new_roi_offset[d] + new_roi_dim[d]) 
	    - (bxf->roi_offset[d] + bxf->roi_offset[d]);

	if (roi_offset_diff[d] < 0) {
	    eb[d] = (bxf->vox_per_rgn[d] - roi_offset_diff[d] - 1) / bxf->vox_per_rgn[d];
	    extend_needed = 1;
	} else {
	    eb[d] = 0;
	}
	if (roi_corner_diff[d] > 0) {
	    ea[d] = (bxf->vox_per_rgn[d] + roi_corner_diff[d] - 1) / bxf->vox_per_rgn[d];
	    extend_needed = 1;
	} else {
	    ea[d] = 0;
	}
    }

    if (extend_needed) {
	/* Allocate new memory */
	for (d = 0; d < 3; d++) {
	    new_rdims[d] = bxf->rdims[d] + ea[d] + eb[d];
	    new_cdims[d] = bxf->cdims[d] + ea[d] + eb[d];
	}
	new_num_knots = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2];
	new_num_coeff = bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2] * 3;
	new_coeff = (float*) malloc (sizeof(float) * new_num_coeff);
	memset (new_coeff, 0, sizeof(float) * new_num_coeff);

	/* Copy coefficients to new memory */
	for (old_idx = 0, k = 0; k < bxf->cdims[2]; k++) {
	    for (j = 0; j < bxf->cdims[1]; j++) {
		for (i = 0; i < bxf->cdims[0]; i++) {
		    int new_idx = 3 * (((((k + eb[2]) * new_cdims[1]) + (j + eb[1])) * new_cdims[0]) + (i + eb[0]));
		    for (d = 0; d < 3; d++, old_idx++, new_idx++) {
			new_coeff[new_idx] = bxf->coeff[old_idx];
		    }
		}
	    }
	}

	/* Free old memory */
	free (bxf->coeff);

	/* Copy over new data into bxf */
	for (d = 0; d < 3; d++) {
	    bxf->rdims[d] = new_rdims[d];
	    bxf->cdims[d] = new_cdims[d];
	}
	bxf->num_knots = new_num_knots;
	bxf->num_coeff = new_num_coeff;
	bxf->coeff = new_coeff;

	/* Special consideration to ROI */
	for (d = 0; d < 3; d++) {
	    bxf->roi_offset[d] = bxf->roi_offset[d] - eb[d] * bxf->vox_per_rgn[d];
	    bxf->roi_dim[d] = new_roi_dim[d] + (new_roi_offset[d] - bxf->roi_offset[d]);
	}
    }
}

static void
bspline_initialize_mi_vol (BSPLINE_MI_Hist_Parms* hparms, Volume* vol)
{
    int i;
    float min_vox, max_vox;
    float* img = (float*) vol->img;

    if (!img) {
	logfile_printf ("Error trying to create histogram from empty image\n");
	exit (-1);
    }
    min_vox = max_vox = img[0];
    for (i = 0; i < vol->npix; i++) {
	if (img[i] < min_vox) {
	    min_vox = img[i];
	} else if (img[i] > max_vox) {
	    max_vox = img[i];
	}
    }

    /* To avoid rounding issues, top and bottom bin are only half full */
    hparms->delta = (max_vox - min_vox) / (hparms->bins - 1);
    hparms->offset = min_vox - 0.5 * hparms->delta;
}

static void
bspline_initialize_mi (BSPLINE_Parms* parms, Volume* fixed, Volume* moving)
{
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    mi_hist->m_hist = (float*) malloc (sizeof (float) * mi_hist->moving.bins);
    mi_hist->f_hist = (float*) malloc (sizeof (float) * mi_hist->fixed.bins);
    mi_hist->j_hist = (float*) malloc (sizeof (float) * mi_hist->fixed.bins * mi_hist->moving.bins);
    bspline_initialize_mi_vol (&mi_hist->moving, moving);
    bspline_initialize_mi_vol (&mi_hist->fixed, fixed);
}

gpuit_EXPORT
void
bspline_xform_free (BSPLINE_Xform* bxf)
{
    free (bxf->coeff);
    free (bxf->q_lut);
    free (bxf->c_lut);
}

gpuit_EXPORT
void
bspline_parms_free (BSPLINE_Parms* parms)
{
    if (parms->ssd.grad) {
	free (parms->ssd.grad);
    }
    if (parms->mi_hist.j_hist) {
	free (parms->mi_hist.f_hist);
	free (parms->mi_hist.m_hist);
	free (parms->mi_hist.j_hist);
    }
}

/* This function will split the amout to add between two bins (linear interp) 
    based on m_val, but one bin based on f_val. */
inline void
bspline_mi_hist_lookup (
	long j_idxs[2],		    /* Output: Joint histogram indices */
	long m_idxs[2],		    /* Output: Moving marginal indices */
	long f_idxs[1],		    /* Output: Fixed marginal indices */
	float fxs[2],		    /* Output: Fraction contribution at indices */
	BSPLINE_MI_Hist* mi_hist,   /* Input:  The histogram */
	float f_val,		    /* Input:  Intensity of fixed image */
	float m_val		    /* Input:  Intensity of moving image */
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
    mf_1 = (midx - midx_trunc) / mi_hist->moving.delta;
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

/* This function will split the amout to add between two bins (linear interp) 
    based on m_val, but one bin based on f_val. */
inline void
bspline_mi_hist_add (
	BSPLINE_MI_Hist* mi_hist,   /* The histogram */
	float f_val,		    /* Intensity of fixed image */
	float m_val,		    /* Intensity of moving image */
	float amt		    /* How much to add to histogram */
)
{
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];

    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_val, m_val);

    fxs[0] *= amt;
    fxs[1] *= amt;

    f_hist[f_idxs[0]] += amt;	    /* This is inefficient */
    m_hist[m_idxs[0]] += fxs[0];
    m_hist[m_idxs[1]] += fxs[1];
    j_hist[j_idxs[0]] += fxs[0];
    j_hist[j_idxs[1]] += fxs[1];
}

/* This algorithm uses a un-normalized score. */
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
	    if (j_hist[v] > hist_thresh) {
		score -= j_hist[v] * logf (fnv * j_hist[v] / (m_hist[j] * f_hist[i]));
	    }
	}
    }

    score = score / fnv;
    return score;
}

inline void
bspline_interp_pix (float out[3], BSPLINE_Xform* bxf, int p[3], int qidx)
{
    int i, j, k, m;
    int cidx;
    float* q_lut = &bxf->q_lut[qidx*64];

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

static void
bspline_interp_pix_b (float out[3], BSPLINE_Xform* bxf, int pidx, int qidx)
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

gpuit_EXPORT
void
bspline_interpolate_vf (Volume* interp, 
			BSPLINE_Xform* bxf)
{
    int i, j, k, v;
    int p[3];
    int q[3];
    float* out;
    float* img = (float*) interp->img;
    int qidx;

    memset (img, 0, interp->npix*3*sizeof(float));
    for (k = 0; k < bxf->roi_dim[2]; k++) {
	p[2] = k / bxf->vox_per_rgn[2];
	q[2] = k % bxf->vox_per_rgn[2];
	for (j = 0; j < bxf->roi_dim[1]; j++) {
	    p[1] = j / bxf->vox_per_rgn[1];
	    q[1] = j % bxf->vox_per_rgn[1];
	    for (i = 0; i < bxf->roi_dim[0]; i++) {
		p[0] = i / bxf->vox_per_rgn[0];
		q[0] = i % bxf->vox_per_rgn[0];
		qidx = q[2] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[0]
			+ q[1] * bxf->vox_per_rgn[0] + q[0];
		v = (k+bxf->roi_offset[2]) * interp->dim[0] * interp->dim[1]
			+ (j+bxf->roi_offset[1]) * interp->dim[0] + (i+bxf->roi_offset[0]);
		out = &img[3*v];
		bspline_interp_pix (out, bxf, p, qidx);
	    }
	}
    }
}

inline void
bspline_update_grad (BSPLINE_Parms* parms, BSPLINE_Xform* bxf, int p[3], int qidx, float dc_dv[3])
{
    BSPLINE_Score* ssd = &parms->ssd;
    int i, j, k, m;
    int cidx;
    float* q_lut = &bxf->q_lut[qidx*64];

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

inline void
bspline_update_grad_b_inline (BSPLINE_Parms* parms, BSPLINE_Xform* bxf, 
		     int pidx, int qidx, float dc_dv[3])
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
		m ++;
	    }
	}
    }
}

void
bspline_update_grad_b (BSPLINE_Parms* parms, BSPLINE_Xform* bxf, 
		     int pidx, int qidx, float dc_dv[3])
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
    *mar = a + ROUND_INT (dxyz[d]);
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
clamp_linear_interpolate_inline (
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
    *mar = ROUND_INT (ma);
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
clamp_linear_interpolate (
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
    *mar = ROUND_INT (ma);
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
clamp_quadratic_interpolate_inline (
    float ma,           /* (Unrounded) pixel coordinate (in vox units) */
    long dmax,		/* Maximum coordinate in this dimension */
    long maqs[3],	/* x, y, or z coord of 3 pixels in moving img */
    float faqs[3]	/* Fraction of interpolant for 3 voxels */
)
{
    float marf = floorf (ma + 0.5);	/* marf = ma, rounded, floating */
    long mari = (long) marf;		/* mari = ma, rounded, integer */

    float t = ma - marf + 0.5;
    float t2 = t * t;
    float t22 = 0.5 * t2;

    faqs[0] = t22;
    faqs[1] = - t2 + t + 0.5;
    faqs[2] = t22 - t + 0.5;

    maqs[0] = mari - 1;
    maqs[1] = mari;
    maqs[2] = mari + 1;

    if (maqs[0] < 0) {
	maqs[0] = 0;
	if (maqs[1] < 0) {
	    maqs[1] = 0;
	    if (maqs[2] < 0) {
		maqs[2] = 0;
	    }
	}
    } else if (maqs[2] >= dmax) {
	maqs[2] = dmax - 1;
	if (maqs[1] >= dmax) {
	    maqs[1] = dmax - 1;
	    if (maqs[0] >= dmax) {
		maqs[0] = dmax - 1;
	    }
	}
    }
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
compute_dS_dP (float* j_hist, float* f_hist, float* m_hist, long* j_idxs, long* f_idxs, long* m_idxs, float num_vox_f, float* fxs, float score, int debug)
{
    float dS_dP_0, dS_dP_1, dS_dP;
    const float j_hist_thresh = 0.0001f;

#if defined (commentout)
#endif
	
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

/* Mutual information version of implementation "C" */
static void
bspline_score_c_mi (BSPLINE_Parms *parms, 
		    BSPLINE_Xform *bxf, 
		    Volume *fixed, 
		    Volume *moving, 
		    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &parms->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    int i;
    int ri, rj, rk;
    int fi, fj, fk, fv;
    float mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int mif, mjf, mkf, mvf;  /* Floor */
    int mir, mjr, mkr;       /* Round */
    long miqs[3], mjqs[3], mkqs[3];	/* Rounded indices */
    float fxqs[3], fyqs[3], fzqs[3];	/* Fractional values */
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float dxyz[3];
    int num_vox;
    float num_vox_f;
    int pidx, qidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    float mse_score = 0.0f;
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
    const float ONE_THIRD = 1.0f / 3.0f;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dump_mi_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    start_clock = clock();

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(float));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(float));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(float));
    num_vox = 0;

    /* PASS 1 - Accumulate histogram */
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

		/* Get B-spline deformation vector */
		pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
		qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);
// 
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

#if defined (commentout)
#endif
		/* Compute linear interpolation fractions */
		clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
		clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
		clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

		/* Compute linearly interpolated moving image value */
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

#if defined (commentout)
		/* LINEAR INTERPOLATION */
		bspline_mi_hist_add (mi_hist, f_img[fv], m_val, 1.0);
#endif

#if defined (commentout)
		/* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], m_x1y1z1);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+1], m_x2y1z1);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[0]], m_x1y2z1);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[0]+1], m_x2y2z1);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]], m_x1y1z2);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+1], m_x2y1z2);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]], m_x1y2z2);
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1], m_x2y2z2);
#endif

		/* Compute quadratic interpolation fractions */
		clamp_quadratic_interpolate_inline (mi, moving->dim[0], miqs, fxqs);
		clamp_quadratic_interpolate_inline (mj, moving->dim[1], mjqs, fyqs);
		clamp_quadratic_interpolate_inline (mk, moving->dim[2], mkqs, fzqs);
#if 0
		printf ("Done! [%d %d %d], [%d %d %d], [%d %d %d]\n",
		    miqs[0], miqs[1], miqs[2],
		    mjqs[0], mjqs[1], mjqs[2],
		    mkqs[0], mkqs[1], mkqs[2]
		    );
#endif

		/* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
		mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * (fxqs[1] + fyqs[1] + fzqs[1]));
		mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[0];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fxqs[0]);
		mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[2];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fxqs[2]);
		mvf = (mkqs[1] * moving->dim[1] + mjqs[0]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fyqs[0]);
		mvf = (mkqs[1] * moving->dim[1] + mjqs[2]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fyqs[2]);
		mvf = (mkqs[0] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fzqs[0]);
		mvf = (mkqs[2] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
		bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], ONE_THIRD * fzqs[2]);

		/* Compute intensity difference */
		diff = f_img[fv] - m_val;
		mse_score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Compute score */
    ssd->score = mi_hist_score (mi_hist, num_vox);
    num_vox_f = (float) num_vox;

    /* PASS 2 - Compute gradient */
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

		/* Get B-spline deformation vector */
		pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
		qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

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

#if defined (commentout)
		/* Compute interpolation fractions */
		clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
		clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
		clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

		mvf = (mkf * moving->dim[1] + mjf) * moving->dim[0] + mif;
		m_x1y1z1 = fx1 * fy1 * fz1;
		m_x2y1z1 = fx2 * fy1 * fz1;
		m_x1y2z1 = fx1 * fy2 * fz1;
		m_x2y2z1 = fx2 * fy2 * fz1;
		m_x1y1z2 = fx1 * fy1 * fz2;
		m_x2y1z2 = fx2 * fy1 * fz2;
		m_x1y2z2 = fx1 * fy2 * fz2;
		m_x2y2z2 = fx2 * fy2 * fz2;

		if (debug) {
		    printf ("m_xyz %g %g %g %g %g %g %g %g\n",
			m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1, 
			m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2);
		}
#endif

		/* Compute pixel contribution to gradient based on histogram change

		   There are eight correspondences between fixed and moving.  
		   Each of these eight correspondences will update 2 histogram bins
		     (other options are possible, but this is my current strategy).

		   First, figure out which histogram bins this correspondence influences
		   by calling bspline_mi_hist_lookup().

		   Next, compute dS/dP * dP/dx. 
		    dP/dx is zero outside of the 8 neighborhood.  Otherwise it is +/- 1/pixel size.
		    dS/dP is 1/N (ln (N P / Pf Pm) - I)
		    dS/dP is weighted by the relative contribution of the 2 histogram bins.

		   dS_dx and dc_dv are really the same thing.  Just different notation.

		   For dc_dv:
		   We do -= instead of += because our optimizer wants to minimize 
		    instead of maximize.
		   The right hand side is - for "left" voxels, and + for "right" voxels.

		   Use a hard threshold on j_hist[j_idxs] to prevent overflow.  This test 
		    should be reconsidered, because it is theoretically unsound.

		   Some trivial speedups for the future:
		    The pixel size component is constant, so we can post-multiply.
		    1/N is constant, so post-multiply.
		*/
		dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

#if defined (commentout)
		/* The below code is for the 8-neighborhood */
		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - dS_dP;
		dc_dv[1] -= - dS_dP;
		dc_dv[2] -= - dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+1]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= + dS_dP;
		dc_dv[1] -= - dS_dP;
		dc_dv[2] -= - dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+moving->dim[0]]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - dS_dP;
		dc_dv[1] -= + dS_dP;
		dc_dv[2] -= - dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+moving->dim[0]+1]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= + dS_dP;
		dc_dv[1] -= + dS_dP;
		dc_dv[2] -= - dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - dS_dP;
		dc_dv[1] -= - dS_dP;
		dc_dv[2] -= + dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+1]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= + dS_dP;
		dc_dv[1] -= - dS_dP;
		dc_dv[2] -= + dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= - dS_dP;
		dc_dv[1] -= + dS_dP;
		dc_dv[2] -= + dS_dP;

		bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1]);
		dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, num_vox_f, fxs, ssd->score, debug);
		dc_dv[0] -= + dS_dP;
		dc_dv[1] -= + dS_dP;
		dc_dv[2] -= + dS_dP;
#endif

		/* Compute quadratic interpolation fractions */
		clamp_quadratic_interpolate_grad_inline (mi, moving->dim[0], miqs, fxqs);
		clamp_quadratic_interpolate_grad_inline (mj, moving->dim[1], mjqs, fyqs);
		clamp_quadratic_interpolate_grad_inline (mk, moving->dim[2], mkqs, fzqs);

		/* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
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

		bspline_update_grad_b_inline (parms, bxf, pidx, qidx, dc_dv);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    mse_score = mse_score / num_vox;

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }

    end_clock = clock();

    logfile_printf ("SCORE: MI %10.8f MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, mse_score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

/* bspline_score_f_mse:
 * This version is similar to version "D" in that it calculates the
 * score tile by tile. For the gradient, it iterates over each
 * control knot, determines which tiles influence the knot, and then
 * sums the total influence from each of those tiles. This implementation
 * allows for greater parallelization on the GPU.
 */
void bspline_score_f_mse (
	BSPLINE_Parms *parms,
	BSPLINE_Xform *bxf,
	Volume *fixed,
	Volume *moving,
	Volume *moving_grad)
{
	BSPLINE_Score* ssd = &parms->ssd;

    int i, j, k, m;
    int fi, fj, fk, fv;
    float mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int mif, mjf, mkf, mvf;  /* Floor */
    int mir, mjr, mkr, mvr;  /* Round */
    int p[3];
    int q[3];
    float diff;
    float* dc_dv;
	float* dc_dv_row;
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx;
	int qidx;
	int cidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    int* c_lut;

	int knot_x, knot_y, knot_z;
	int tile_x, tile_y, tile_z;
	int x_offset, y_offset, z_offset;
	int total_vox_per_rgn = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
		sprintf (debug_fn, "dump_mse_%02d.txt", it++);
		fp = fopen (debug_fn, "w");
    }

    start_clock = clock();

	// Allocate memory for the dc_dv array. In this implementation, all of the dc_dv values
	// are computed and stored before the gradient calculation.
	dc_dv = (float*)malloc(3 * total_vox_per_rgn * bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] * sizeof(float));

	ssd->score = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    // Serial across tiles
    for (p[2] = 0; p[2] < bxf->rdims[2]; p[2]++) {
		for (p[1] = 0; p[1] < bxf->rdims[1]; p[1]++) {
			for (p[0] = 0; p[0] < bxf->rdims[0]; p[0]++) {

				// Compute linear index for tile
				pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];

				// Find c_lut row for this tile
				c_lut = &bxf->c_lut[pidx*64];

				//logfile_printf ("Kernel 1, tile %d %d %d\n", p[0], p[1], p[2]);

				// Parallel across offsets within the tile
				for (q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
					for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
						for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {

							// Compute linear index for this offset
							qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];

							// Tentatively mark this pixel as no contribution
							dc_dv[(pidx * 3 * total_vox_per_rgn) + (3*qidx+0)] = 0.f;
							dc_dv[(pidx * 3 * total_vox_per_rgn) + (3*qidx+1)] = 0.f;
							dc_dv[(pidx * 3 * total_vox_per_rgn) + (3*qidx+2)] = 0.f;

							// Get (i,j,k) index of the voxel
							fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
							fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
							fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];

							// Some of the pixels are outside image
							if (fi >= bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
							if (fj >= bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
							if (fk >= bxf->roi_offset[2] + bxf->roi_dim[2]) continue;

							//if (p[2] == 4) {
		    				//	logfile_printf ("Kernel 1, pix %d %d %d\n", fi, fj, fk);
		    				//	logfile_printf ("Kernel 1, offset %d %d %d\n", q[0], q[1], q[2]);
							//}

							// Compute physical coordinates of fixed image voxel
							fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;
							fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
							fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;

							// Compute linear index of fixed image voxel
							fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

							// Get B-spline deformation vector
							bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

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

							// Compute interpolation fractions
							clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
							clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
							clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

							// Compute moving image intensity using linear interpolation
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

							// Compute intensity difference
							diff = f_img[fv] - m_val;

							// We'll go ahead and accumulate the score here, but you would 
							// have to reduce somewhere else instead.
							ssd->score += diff * diff;
							num_vox++;

							// Compute spatial gradient using nearest neighbors
							mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

							// Store dc_dv for this offset
							dc_dv_row = &dc_dv[3 * total_vox_per_rgn * pidx];
							dc_dv_row[3*qidx+0] = diff * m_grad[3*mvr+0];  // x component
							dc_dv_row[3*qidx+1] = diff * m_grad[3*mvr+1];  // y component
							dc_dv_row[3*qidx+2] = diff * m_grad[3*mvr+2];  // z component
						}
					}
				}
			}
		}
	}

	// Iterate through each of the control knots.
	for(cidx = 0; cidx < bxf->num_knots; cidx++) {

		// Determine the x, y, and z offset of the knot within the grid.
		knot_x = cidx % bxf->cdims[0];
		knot_y = ((cidx - knot_x) / bxf->cdims[0]) % bxf->cdims[1];
		knot_z = ((((cidx - knot_x) / bxf->cdims[0]) - knot_y) / bxf->cdims[1]) % bxf->cdims[2];

		// Subtract 1 from each of the knot indices to account for the differing origin
		// between the knot grid and the tile grid.
		knot_x -= 1;
		knot_y -= 1;
		knot_z -= 1;

		// Iterate through each of the 64 tiles that influence this control knot.
		for(z_offset = -2; z_offset < 2; z_offset++) {
			for(y_offset = -2; y_offset < 2; y_offset++) {
				for(x_offset = -2; x_offset < 2; x_offset++) {

					// Using the current x, y, and z offset from the control knot position,
					// calculate the index for one of the tiles that influence this knot.
					tile_x = knot_x + x_offset;
					tile_y = knot_y + y_offset;
					tile_z = knot_z + z_offset;

					// Determine if the tile lies within the volume.
					if((tile_x >= 0 && tile_x < bxf->rdims[0]) &&
						(tile_y >= 0 && tile_y < bxf->rdims[1]) &&
						(tile_z >= 0 && tile_z < bxf->rdims[2])) {
						
						// Compute linear index for tile.
						pidx = ((tile_z * bxf->rdims[1] + tile_y) * bxf->rdims[0]) + tile_x;

						// Find c_lut row for this tile.
						c_lut = &bxf->c_lut[64*pidx];

						// Pull out the dc_dv values for just this tile.
						dc_dv_row = &dc_dv[3 * total_vox_per_rgn * pidx];

						// Find the coefficient index in the c_lut row in order to determine
						// the linear index of the control point with respect to the current tile.
						for(m = 0; m < 64; m++) {
							if(c_lut[m] == cidx) {
								break;
							}
						}

						// Iterate through each of the offsets within the tile and 
						// accumulate the gradient.
						for(qidx = 0, q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
							for(q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
								for(q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++, qidx++) {

									// Find q_lut row for this offset.
									float* q_lut = &bxf->q_lut[64*qidx];							

									// Accumulate update to gradient for this control point.
									ssd->grad[3*cidx+0] += dc_dv_row[3*qidx+0] * q_lut[m];
									ssd->grad[3*cidx+1] += dc_dv_row[3*qidx+1] * q_lut[m];
									ssd->grad[3*cidx+2] += dc_dv_row[3*qidx+2] * q_lut[m];
								}
							}
						}
					}
				}
			}
		}
	}

	free (dc_dv);

	if (parms->debug) {
		fclose (fp);
	}

	//dump_coeff (bxf, "coeff_cpu.txt");

	/* Normalize score for MSE */
	ssd->score = ssd->score / num_vox;
	for (i = 0; i < bxf->num_coeff; i++) {
		ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
	}

	//dump_gradient(bxf, ssd, "grad_cpu.txt");

	/* Normalize gradient */
	ssd_grad_norm = 0;
	ssd_grad_mean = 0;
	for (i = 0; i < bxf->num_coeff; i++) {
		ssd_grad_mean += ssd->grad[i];
		ssd_grad_norm += fabs (ssd->grad[i]);
	}

	end_clock = clock();

	printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
		ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
		(double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

void bspline_score_e_mse (
			  BSPLINE_Parms *parms, 
			  BSPLINE_Xform* bxf, 
			  Volume *fixed, 
			  Volume *moving, 
			  Volume *moving_grad)
{
    BSPLINE_Score* ssd = &parms->ssd;
    int i, j, k, m;
    int fi, fj, fk, fv;
    float mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int mif, mjf, mkf, mvf;  /* Floor */
    int mir, mjr, mkr, mvr;  /* Round */
    int s[3];
    int p[3];
    int q[3];
    float diff;
    float* dc_dv;
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    int cidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    int* c_lut;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    start_clock = clock();

    dc_dv = (float*) malloc (3*bxf->vox_per_rgn[0]*bxf->vox_per_rgn[1]*bxf->vox_per_rgn[2]*sizeof(float));
    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
  	
    // There are 64 sets of tiles for which the score can be calculated in parallel.  Iterate through these sets.
    for(s[2] = 0; s[2] < 4; s[2]++) {
	for(s[1] = 0; s[1] < 4; s[1]++) {
	    for(s[0] = 0; s[0] < 4; s[0]++) {
	
		// Run kernel #1 and kernel #2 here.

		// Calculate the score for each of the tiles in the set in serial.
		for(p[2] = s[2]; p[2] < bxf->rdims[2]; p[2] += 4) {
		    for(p[1] = s[1]; p[1] < bxf->rdims[1]; p[1] += 4) {
			for(p[0] = s[0]; p[0] < bxf->rdims[0]; p[0] += 4) {

			    // printf("Tile: (%d, %d, %d)\n", p[0], p[1], p[2]);

			    // Compute linear index for tile.
			    pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];

			    // Find c_lut row for this tile.
			    c_lut = &bxf->c_lut[pidx*64];

			    // Parallel across the offsets within the tile.
			    for (q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
				for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
				    for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
		    
					// Compute the linear index for this offset.
					qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];
									    
					// Tentatively mark this pixel as no contribution.
					dc_dv[3*qidx+0] = 0.f;
					dc_dv[3*qidx+1] = 0.f;
					dc_dv[3*qidx+2] = 0.f;
									    
					// Get (i,j,k) index of the voxel.
					fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
					fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
					fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];
									    
					// Some of the pixels are outside image.
					if (fi > bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
					if (fj > bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
					if (fk > bxf->roi_offset[2] + bxf->roi_dim[2]) continue;
									    
					// Compute physical coordinates of fixed image voxel.
					fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;
					fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
					fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;
									    
					// Compute linear index of fixed image voxel.
					fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;
									    
					// Get B-spline deformation vector.
					bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);
									    
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

					// We'll go ahead and accumulate the score here, but you would 
					// have to reduce somewhere else instead.
					ssd->score += diff * diff;
					num_vox ++;

					// Compute spatial gradient using nearest neighbors.
					mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

					// Store dc_dv for this offset.
					dc_dv[3*qidx+0] = diff * m_grad[3*mvr+0];  // x component
					dc_dv[3*qidx+1] = diff * m_grad[3*mvr+1];  // y component
					dc_dv[3*qidx+2] = diff * m_grad[3*mvr+2];  // z component
				    }
				}
			    }

			    // Parallel across 64 control points.
			    for (k = 0; k < 4; k++) {
				for (j = 0; j < 4; j++) {
				    for (i = 0; i < 4; i++) {

					// Compute linear index of control point.
					m = k*16 + j*4 + i;

					// Find index of control point within coefficient array.
					cidx = c_lut[m] * 3;

					// Serial across offsets within kernel.
					for (qidx = 0, q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
					    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
						for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++, qidx++) {

						    // Find q_lut row for this offset.
						    float* q_lut = &bxf->q_lut[qidx*64];

						    // Accumulate update to gradient for this control point.
						    ssd->grad[cidx+0] += dc_dv[3*qidx+0] * q_lut[m];
						    ssd->grad[cidx+1] += dc_dv[3*qidx+1] * q_lut[m];
						    ssd->grad[cidx+2] += dc_dv[3*qidx+2] * q_lut[m];
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
	}
    }

    free (dc_dv);

    if (parms->debug) {
	fclose (fp);
    }

    //dump_coeff (bxf, "coeff_cpu.txt");

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    //dump_gradient(bxf, ssd, "grad_cpu.txt");

    /* Normalize gradient */
    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }

    end_clock = clock();

    printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}


/* Mean-squared error version of implementation "D" */
/* This design could be useful for a low-memory GPU implementation

for each tile in serial

  for each offset in parallel
    for each control point in serial
      accumulate into dxyz
    compute diff
    store dc_dv for this offset

  for each control point in parallel
    for each offset in serial
      accumulate into coefficient

end

The dual design which I didn't use:

for each offset in serial

  for each tile in parallel
    for each control point in serial
      accumulate into dxyz
    compute diff
    store dc_dv for this tile

  for each control point in parallel
    for each offset in serial
      accumulate into coefficient

end
*/

void
bspline_score_d_mse (
		 BSPLINE_Parms *parms, 
		 BSPLINE_Xform* bxf, 
		 Volume *fixed, 
		 Volume *moving, 
		 Volume *moving_grad)
{
    BSPLINE_Score* ssd = &parms->ssd;
    int i, j, k, m;
    int fi, fj, fk, fv;
    float mi, mj, mk;
    float fx, fy, fz;
    float mx, my, mz;
    int mif, mjf, mkf, mvf;  /* Floor */
    int mir, mjr, mkr, mvr;  /* Round */
    int p[3];
    int q[3];
    float diff;
    float* dc_dv;
    float fx1, fx2, fy1, fy2, fz1, fz2;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    int cidx;
    float ssd_grad_norm, ssd_grad_mean;
    clock_t start_clock, end_clock;
    float m_val;
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    int* c_lut;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    start_clock = clock();

    dc_dv = (float*) malloc (3*bxf->vox_per_rgn[0]*bxf->vox_per_rgn[1]*bxf->vox_per_rgn[2]*sizeof(float));
    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    /* Serial across tiles */
    for (p[2] = 0; p[2] < bxf->rdims[2]; p[2]++) {
		for (p[1] = 0; p[1] < bxf->rdims[1]; p[1]++) {
			for (p[0] = 0; p[0] < bxf->rdims[0]; p[0]++) {

				/* Compute linear index for tile */
				pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];

				/* Find c_lut row for this tile */
				c_lut = &bxf->c_lut[pidx*64];

				//logfile_printf ("Kernel 1, tile %d %d %d\n", p[0], p[1], p[2]);

				/* Parallel across offsets */
				for (q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
					for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
						for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {

							/* Compute linear index for this offset */
							qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];

							/* Tentatively mark this pixel as no contribution */
							dc_dv[3*qidx+0] = 0.f;
							dc_dv[3*qidx+1] = 0.f;
							dc_dv[3*qidx+2] = 0.f;

							/* Get (i,j,k) index of the voxel */
							fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
							fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
							fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];

							/* Some of the pixels are outside image */
							if (fi >= bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
							if (fj >= bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
							if (fk >= bxf->roi_offset[2] + bxf->roi_dim[2]) continue;

							//if (p[2] == 4) {
		    				//	logfile_printf ("Kernel 1, pix %d %d %d\n", fi, fj, fk);
		    				//	logfile_printf ("Kernel 1, offset %d %d %d\n", q[0], q[1], q[2]);
							//}

							/* Compute physical coordinates of fixed image voxel */
							fx = bxf->img_origin[0] + bxf->img_spacing[0] * fi;
							fy = bxf->img_origin[1] + bxf->img_spacing[1] * fj;
							fz = bxf->img_origin[2] + bxf->img_spacing[2] * fk;

							/* Compute linear index of fixed image voxel */
							fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

							/* Get B-spline deformation vector */
							bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

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
							clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
							clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
							clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

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

							/* We'll go ahead and accumulate the score here, but you would 
							have to reduce somewhere else instead */
							ssd->score += diff * diff;
							num_vox ++;

							/* Compute spatial gradient using nearest neighbors */
							mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

							/* Store dc_dv for this offset */
							dc_dv[3*qidx+0] = diff * m_grad[3*mvr+0];  /* x component */
							dc_dv[3*qidx+1] = diff * m_grad[3*mvr+1];  /* y component */
							dc_dv[3*qidx+2] = diff * m_grad[3*mvr+2];  /* z component */
						}
					}
				}

				//logfile_printf ("Kernel 2, tile %d %d %d\n", p[0], p[1], p[2]);

				/* Parallel across 64 control points */
				for (k = 0; k < 4; k++) {
					for (j = 0; j < 4; j++) {
						for (i = 0; i < 4; i++) {

							/* Compute linear index of control point */
							m = k*16 + j*4 + i;

							/* Find index of control point within coefficient array */
							cidx = c_lut[m] * 3;

							/* Serial across offsets within kernel */
							for (qidx = 0, q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
								for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
									for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++, qidx++) {

										/* Find q_lut row for this offset */
										float* q_lut = &bxf->q_lut[qidx*64];

										/* Accumulate update to gradient for this 
											control point */
										ssd->grad[cidx+0] += dc_dv[3*qidx+0] * q_lut[m];
										ssd->grad[cidx+1] += dc_dv[3*qidx+1] * q_lut[m];
										ssd->grad[cidx+2] += dc_dv[3*qidx+2] * q_lut[m];

									}
								}
							}
						}
					}
				}
			}
		}
    }
    free (dc_dv);

    if (parms->debug) {
	fclose (fp);
    }

    //dump_coeff (bxf, "coeff.txt");

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    /* Normalize gradient */
    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }

    end_clock = clock();

	fp = fopen("scores.txt", "a+");
	fprintf(fp, "%f\n", ssd->score);
	fclose(fp);

    logfile_printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

/* Mean-squared error version of implementation "C" */
/* Implementation "C" is slower than "B", but yields a smoother cost function 
   for use by L-BFGS-B.  It uses linear interpolation of moving image, 
   and nearest neighbor interpolation of gradient */
void
bspline_score_c_mse (
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

    if (parms->debug) {
	sprintf (debug_fn, "dump_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    start_clock = clock();

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

		/* Get B-spline deformation vector */
		pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
		qidx = ((q[2] * bxf->vox_per_rgn[1] + q[1]) * bxf->vox_per_rgn[0]) + q[0];
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

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
		clamp_linear_interpolate_inline (mi, moving->dim[0]-1, &mif, &mir, &fx1, &fx2);
		clamp_linear_interpolate_inline (mj, moving->dim[1]-1, &mjf, &mjr, &fy1, &fy2);
		clamp_linear_interpolate_inline (mk, moving->dim[2]-1, &mkf, &mkr, &fz1, &fz2);

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
		bspline_update_grad_b_inline (parms, bxf, pidx, qidx, dc_dv);
		
		if (parms->debug) {
		    fprintf (fp, "%d %d %d %g %g %g\n", ri, rj, rk, dc_dv[0], dc_dv[1], dc_dv[2]);
		}

		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    //dump_coeff (bxf, "coeff.txt");

    /* Normalize score for MSE */
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

    end_clock = clock();
#if defined (commentout)
    printf ("Single iteration CPU [b] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
    printf ("NUM_VOX = %d\n", num_vox);
    printf ("MSE = %g\n", ssd->score);
    printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    printf ("GRAD_NORM = %g\n", ssd_grad_norm);
#endif
    logfile_printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

/* This is the fastest known version.  It does nearest neighbors 
   interpolation of both moving image and gradient which doesn't 
   work with stock L-BFGS-B optimizer. */
void
bspline_score_b_mse (BSPLINE_Parms *parms, 
		 BSPLINE_Xform *bxf, 
		 Volume *fixed, 
		 Volume *moving, 
		 Volume *moving_grad)
{
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

		/* Get B-spline deformation vector */
		pidx = ((p[2] * bxf->rdims[1] + p[1]) * bxf->rdims[0]) + p[0];
		qidx = q[2] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[0]
			+ q[1] * bxf->vox_per_rgn[0] + q[0];
		bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->pix_spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->pix_spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->pix_spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;
		mv = (mk * moving->dim[1] + mj) * moving->dim[0] + mi;

		/* Compute intensity difference */
		diff = f_img[fv] - m_img[mv];

		/* Compute spatial gradient using nearest neighbors */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */

		bspline_update_grad_b (parms, bxf, pidx, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = ssd->grad[i] / num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }

    end_clock = clock();
#if defined (commentout)
    logfile_printf ("Single iteration CPU [b] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
    logfile_printf ("NUM_VOX = %d\n", num_vox);
    logfile_printf ("MSE = %g\n", ssd->score);
    logfile_printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    logfile_printf ("GRAD_NORM = %g\n", ssd_grad_norm);
#endif
    logfile_printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

void
bspline_score_a_mse (BSPLINE_Parms *parms, 
		 BSPLINE_Xform* bxf, 
		 Volume *fixed, 
		 Volume *moving, 
		 Volume *moving_grad
		 )
{
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

		/* Get B-spline deformation vector */
		qidx = q[2] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[0]
			+ q[1] * bxf->vox_per_rgn[0] + q[0];
		bspline_interp_pix (dxyz, bxf, p, qidx);

		/* Compute coordinate of fixed image voxel */
		fv = fk * fixed->dim[0] * fixed->dim[1] + fj * fixed->dim[0] + fi;

		/* Find correspondence in moving image */
		mx = fx + dxyz[0];
		mi = ROUND_INT ((mx - moving->offset[0]) / moving->pix_spacing[0]);
		if (mi < 0 || mi >= moving->dim[0]) continue;
		my = fy + dxyz[1];
		mj = ROUND_INT ((my - moving->offset[1]) / moving->pix_spacing[1]);
		if (mj < 0 || mj >= moving->dim[1]) continue;
		mz = fz + dxyz[2];
		mk = ROUND_INT ((mz - moving->offset[2]) / moving->pix_spacing[2]);
		if (mk < 0 || mk >= moving->dim[2]) continue;
		mv = (mk * moving->dim[1] + mj) * moving->dim[0] + mi;

		/* Compute intensity difference */
		diff = f_img[fv] - m_img[mv];

		/* Compute spatial gradient using nearest neighbors */
		dc_dv[0] = diff * m_grad[3*mv+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mv+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mv+2];  /* z component */
		bspline_update_grad (parms, bxf, p, qidx, dc_dv);
		
		ssd->score += diff * diff;
		num_vox ++;
	    }
	}
    }

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] /= num_vox;
    }

    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_mean += ssd->grad[i];
	ssd_grad_norm += fabs (ssd->grad[i]);
    }
    end_clock = clock();
#if defined (commentout)
    logfile_printf ("Single iteration CPU [a] = %f seconds\n", 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);

    logfile_printf ("MSE = %g\n", ssd->score);
    logfile_printf ("GRAD_MEAN = %g\n", ssd_grad_mean);
    logfile_printf ("GRAD_NORM = %g\n", ssd_grad_norm);
#endif
    logfile_printf ("SCORE: MSE %6.3f NV [%6d] GM %6.3f GN %6.3f [%6.3f secs]\n", 
	    ssd->score, num_vox, ssd_grad_mean, ssd_grad_norm, 
	    (double)(end_clock - start_clock)/CLOCKS_PER_SEC);
}

void
bspline_score (BSPLINE_Parms *parms, 
	       BSPLINE_Xform* bxf, 
	       Volume *fixed, 
	       Volume *moving, 
	       Volume *moving_grad)
{
#if (HAVE_BROOK) && (BUILD_BSPLINE_BROOK)
    if (parms->threading == BTHR_BROOK) {
	printf("Using Brook GPU. \n");
	bspline_score_on_gpu_reference (parms, fixed, moving, moving_grad);
	return;
    }
#endif

#if (HAVE_CUDA)
    if (parms->threading == BTHR_CUDA) {
	logfile_printf("Using CUDA.\n");
	switch (parms->implementation) {
	case 'c':
	    bspline_cuda_score_c_mse(parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'd':
	    bspline_cuda_score_d_mse(parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'e':
	    bspline_cuda_score_e_mse_v2(parms, bxf, fixed, moving, moving_grad);
	    //bspline_cuda_score_e_mse(parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'f':
	    bspline_cuda_score_f_mse(parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'g':
	    bspline_cuda_score_g_mse(parms, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_cuda_score_f_mse(parms, bxf, fixed, moving, moving_grad);
	    break;
	}
	return;
    }
#endif

    if (parms->metric == BMET_MSE) {
	logfile_printf ("Using CPU. \n");
	switch (parms->implementation) {
	case 'a':
	    bspline_score_a_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'b':
	    bspline_score_b_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'c':
	    bspline_score_c_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'd':
	    bspline_score_d_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'e':
	    bspline_score_e_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	case 'f':
	    bspline_score_f_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_score_c_mse (parms, bxf, fixed, moving, moving_grad);
	    break;
	}
    } else {
	bspline_score_c_mi (parms, bxf, fixed, moving, moving_grad);
    }
}

void
bspline_optimize_steepest (
		BSPLINE_Xform *bxf, 
		BSPLINE_Parms *parms, 
		Volume *fixed, 
		Volume *moving, 
		Volume *moving_grad
		)
{
    BSPLINE_Score* ssd = &parms->ssd;
    int i, it;
//    float a = 0.003f;
//    float alpha = 0.5f, A = 10.0f;
    float a, gamma;
    float gain = 1.5;
    float ssd_grad_norm;
    float old_score;

    /* Get score and gradient */
    bspline_score (parms, bxf, fixed, moving, moving_grad);
    old_score = parms->ssd.score;

    /* Set alpha based on norm gradient */
    ssd_grad_norm = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_norm += fabs (ssd->grad[i]);
    }
    a = 1.0f / ssd_grad_norm;
    gamma = a;
    logfile_printf ("Initial gamma is %g\n", gamma);

    /* Give a little feedback to the user */
    bspline_display_coeff_stats (bxf);

    for (it = 0; it < parms->max_its; it++) {
	char fn[128];

	logfile_printf ("Beginning iteration %d, gamma = %g\n", it, gamma);

	/* Save some debugging information */
	if (parms->debug) {
	    if (parms->metric == BMET_MI) {
		sprintf (fn, "grad_mi_%02d.txt", it);
	    } else {
		sprintf (fn, "grad_mse_%02d.txt", it);
	    }
	    dump_gradient (bxf, ssd, fn);
	    if (parms->metric == BMET_MI) {
		sprintf (fn, "hist_%02d.txt", it);
		dump_hist (&parms->mi_hist, fn);
	    }
	}

	/* Update coefficients */
	//gamma = a / pow(it + A, alpha);
	for (i = 0; i < bxf->num_coeff; i++) {
	    bxf->coeff[i] = bxf->coeff[i] + gamma * ssd->grad[i];
	}

	/* Get score and gradient */
	bspline_score (parms, bxf, fixed, moving, moving_grad);

	/* Update gamma */
	if (parms->ssd.score < old_score) {
	    gamma *= gain;
	} else {
	    gamma /= gain;
	}
	old_score = parms->ssd.score;

	/* Give a little feedback to the user */
	bspline_display_coeff_stats (bxf);
    }
}

gpuit_EXPORT
void
bspline_optimize (BSPLINE_Xform* bxf, 
		  BSPLINE_Parms *parms, 
		  Volume *fixed, 
		  Volume *moving, 
		  Volume *moving_grad)
{
    /* GCS FIX: This is a terrible way to handle gradient.  Should be separated 
	from parms?  */
    /* Make sure gradient is allocated */
    if (parms->ssd.grad) {
	free (parms->ssd.grad);
    }
    parms->ssd.grad = (float*) malloc (bxf->num_coeff * sizeof(float));
    memset (parms->ssd.grad, 0, bxf->num_coeff * sizeof(float));

    log_parms (parms);
    log_bxf_header (bxf);

    if (parms->metric == BMET_MI) {
	bspline_initialize_mi (parms, fixed, moving);
    }

    if (parms->optimization == BOPT_LBFGSB) {
#if defined (HAVE_F2C_LIBRARY)
	bspline_optimize_lbfgsb (bxf, parms, fixed, moving, moving_grad);
#else
	logfile_printf (
	    "LBFGSB not compiled for this platform (f2c library missing).\n"
	    "Reverting to steepest descent.\n"
	    );
	bspline_optimize_steepest (bxf, parms, fixed, moving, moving_grad);
#endif
    } else {
	bspline_optimize_steepest (bxf, parms, fixed, moving, moving_grad);
    }
}
