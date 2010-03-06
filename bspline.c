/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------

    B-Spline basics:
	http://en.wikipedia.org/wiki/B-spline
	http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/surface/bspline-construct.html
	http://graphics.idav.ucdavis.edu/education/CAGDNotes/Quadratic-B-Spline-Surface-Refinement/Quadratic-B-Spline-Surface-Refinement.html

    Proposed variable naming guide:
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
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bspline.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "readmha.h"
#include "timer.h"
#include "volume.h"
#include "xpm.h"

// Fix for logf() under MSVC 2005 32-bit (math.h has an erronous semicolon)
// http://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=98751
#if !defined (_M_IA64) && !defined (_M_AMD64) && defined (_WIN32)
#undef logf
#define logf(x)     ((float)log((double)(x)))
#endif

/* -----------------------------------------------------------------------
   Macros
   ----------------------------------------------------------------------- */
#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])

#define COORDS_FROM_INDEX(ijk, idx, dim) \
	ijk[2] = idx / (dim[0] * dim[1]);	\
	ijk[1] = (idx_tile - (ijk[2] * dim[0] * dim[1])) / dim[0];	\
	ijk[0] = idx_tile - ijk[2] * dim[0] * dim[1] - (ijk[1] * dim[0]);


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

void
bspline_xform_set_default (BSPLINE_Xform* bxf)
{
    int d;

    memset (bxf, 0, sizeof (BSPLINE_Xform));

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

Bspline_state *
bspline_state_create (BSPLINE_Xform *bxf)
{
    Bspline_state *bst = (Bspline_state*) malloc (sizeof (Bspline_state));
    memset (bst, 0, sizeof (Bspline_state));
    bst->ssd.grad = (float*) malloc (bxf->num_coeff * sizeof(float));
    memset (bst->ssd.grad, 0, bxf->num_coeff * sizeof(float));
    return bst;
}

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
    rc = fscanf (fp, "img_origin = %f %f %f\n", 
	&img_origin[0], &img_origin[1], &img_origin[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_origin): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "img_spacing = %f %f %f\n", 
	&img_spacing[0], &img_spacing[1], &img_spacing[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_spacing): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "img_dim = %d %d %d\n", 
	&img_dim[0], &img_dim[1], &img_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_dim): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "roi_offset = %d %d %d\n", 
	&roi_offset[0], &roi_offset[1], &roi_offset[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (roi_offset): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "roi_dim = %d %d %d\n", 
	&roi_dim[0], &roi_dim[1], &roi_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (roi_dim): %s\n", filename);
	goto free_exit;
    }
    rc = fscanf (fp, "vox_per_rgn = %d %d %d\n", 
	&vox_per_rgn[0], &vox_per_rgn[1], &vox_per_rgn[2]);
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
    fclose (fp);
    free (bxf);
    return 0;
}



////////////////////////////////////////////////////////////////////////////////
// FUNCTION: calc_offsets()
//
// This function accepts the number or voxels per control region
// and the dimensions of the control grid to generate where the linear
// memory offsets lie for the beginning of each tile in a 32-byte
// aligned tile-major data organization scheme (such as that which
// is produced by kernel_row_to_tile_major().
//
// Author: James Shackleford
// Data: July 30th, 2009
////////////////////////////////////////////////////////////////////////////////
int* calc_offsets(int* tile_dims, int* cdims)
{
    int vox_per_tile = (tile_dims[0] * tile_dims[1] * tile_dims[2]);
    int pad = 32 - (vox_per_tile % 32);
    int num_tiles = (cdims[0]-3)*(cdims[1]-3)*(cdims[2]-3);

    int* output = (int*)malloc(num_tiles*sizeof(int));

    int i;
    for(i = 0; i < num_tiles; i++)
	output[i] = (vox_per_tile + pad) * i;

    return output;
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: find_knots()
//
// This function takes a tile index as an input and generates
// the indicies of the 64 control knots that it affects.
//
// Returns:
//    knots[idx] - idx is [0,63] and knots[idx] = the linear knot index
//                 of affected control knot # idx within the entire control
//                 knot grid.
//
//    tile_pos[
//
// Author: James Shackleford
// Data: July 13th, 2009
////////////////////////////////////////////////////////////////////////////////
void find_knots(int* knots, int tile_num, int* rdims, int* cdims)
{
    int tile_loc[3];
    int i, j, k;
    int idx = 0;
    int num_tiles_x = cdims[0] - 3;
    int num_tiles_y = cdims[1] - 3;
    int num_tiles_z = cdims[2] - 3;
	
    // First get the [x,y,z] coordinate of
    // the tile in the control grid.
    tile_loc[0] = tile_num % num_tiles_x;
    tile_loc[1] = ((tile_num - tile_loc[0]) / num_tiles_x) % num_tiles_y;
    tile_loc[2] = ((((tile_num - tile_loc[0]) / num_tiles_x) / num_tiles_y) % num_tiles_z);
    /*
      tile_loc[0] = tile_num % rdims[0];
      tile_loc[1] = ((tile_num - tile_loc[0]) / rdims[0]) % rdims[1];
      tile_loc[2] = ((((tile_num - tile_loc[0]) / rdims[0]) / rdims[1]) % rdims[2]);
    */

    // Tiles do not start on the edges of the grid, so we
    // push them to the center of the control grid.
    tile_loc[0]++;
    tile_loc[1]++;
    tile_loc[2]++;

    // Find 64 knots' [x,y,z] coordinates
    // and convert into a linear knot index
    for (k = -1; k < 3; k++)
	for (j = -1; j < 3; j++)
	    for (i = -1; i < 3; i++)
	    {
		knots[idx++] = (cdims[0]*cdims[1]*(tile_loc[2]+k)) + (cdims[0]*(tile_loc[1]+j)) + (tile_loc[0]+i);
	    }

}
////////////////////////////////////////////////////////////////////////////////



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
    logfile_printf ("                    "
		    "CMIN %6.2f CAVG %6.2f CMAX %6.2f\n", 
		    cf_min, cf_avg / bxf->num_coeff, cf_max);
}

void
bspline_save_debug_state (
    BSPLINE_Parms *parms, 
    Bspline_state *bst, 
    BSPLINE_Xform* bxf
)
{
    char fn[1024];

    if (parms->debug) {
	if (parms->metric == BMET_MI) {
	    sprintf (fn, "grad_mi_%02d.txt", bst->it);
	} else {
	    sprintf (fn, "grad_mse_%02d.txt", bst->it);
	}
	dump_gradient (bxf, &bst->ssd, fn);

	sprintf (fn, "coeff_%02d.txt", bst->it);
	dump_coeff (bxf, fn);

	if (parms->metric == BMET_MI) {
	    sprintf (fn, "hist_%02d.txt", bst->it);
	    dump_hist (&parms->mi_hist, fn);
	}
    }
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
void
bspline_xform_initialize 
(
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
    if (!bxf->q_lut) {
	print_and_exit ("Error allocating memory for q_lut\n");
    }
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

    logfile_printf ("rdims = (%d,%d,%d)\n", 
	bxf->rdims[0], bxf->rdims[1], bxf->rdims[2]);
    logfile_printf ("vox_per_rgn = (%d,%d,%d)\n", 
	bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("cdims = (%d %d %d)\n", 
	bxf->cdims[0], bxf->cdims[1], bxf->cdims[2]);
}

/* This extends the bspline grid.  Note, that the new roi_offset 
    in the bxf will not be the same as the one requested, because 
    bxf routines implicitly require that the first voxel of the 
    ROI matches the position of the control point. */
/* GCS -- Is there an implicit assumption that the roi_origin > 0? */
void
bspline_xform_extend (
    BSPLINE_Xform* bxf,	     /* Output: bxf is initialized */
    int new_roi_offset[3],   /* Position of first vox in ROI (in vox) */
    int new_roi_dim[3]	     /* Dimension of ROI (in vox) */
)
{
    int d;
    int roi_offset_diff[3];
    int roi_corner_diff[3];
    int eb[3];  /* # of control points to "extend before" existing grid */
    int ea[3];  /* # of control points to "extend after" existing grid */
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
	    eb[d] = (bxf->vox_per_rgn[d] - roi_offset_diff[d] - 1) 
		/ bxf->vox_per_rgn[d];
	    extend_needed = 1;
	} else {
	    eb[d] = 0;
	}
	if (roi_corner_diff[d] > 0) {
	    ea[d] = (bxf->vox_per_rgn[d] + roi_corner_diff[d] - 1) 
		/ bxf->vox_per_rgn[d];
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

void
bspline_xform_free (BSPLINE_Xform* bxf)
{
    free (bxf->coeff);
    free (bxf->q_lut);
    free (bxf->c_lut);
}

void
bspline_parms_free (BSPLINE_Parms* parms)
{
    if (parms->mi_hist.j_hist) {
	free (parms->mi_hist.f_hist);
	free (parms->mi_hist.m_hist);
	free (parms->mi_hist.j_hist);
    }
}

void
bspline_state_free (Bspline_state* bst)
{
    if (bst->ssd.grad) {
	free (bst->ssd.grad);
    }
    free (bst);
}

/* This function will split the amout to add between two bins (linear interp) 
    based on m_val, but one bin based on f_val. */
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

/* This function will split the amout to add between two bins (linear interp) 
    based on m_val, but one bin based on f_val. */
inline void
bspline_mi_hist_add (
    BSPLINE_MI_Hist* mi_hist,   /* The histogram */
    float f_val,		/* Intensity of fixed image */
    float m_val,		/* Intensity of moving image */
    float amt		        /* How much to add to histogram */
)
{
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];

    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
			    mi_hist, f_val, m_val);

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
bspline_interp_pix_b_inline (
    float out[3], 
    BSPLINE_Xform* bxf, 
    int pidx, 
    int qidx
)
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
bspline_update_grad (Bspline_state *bst, 
		     BSPLINE_Xform* bxf, 
		     int p[3], int qidx, float dc_dv[3])
{
    BSPLINE_Score* ssd = &bst->ssd;
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
bspline_update_grad_b_inline (Bspline_state* bst, BSPLINE_Xform* bxf, 
		     int pidx, int qidx, float dc_dv[3])
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
		m ++;
	    }
	}
    }
}

void
bspline_update_grad_b (Bspline_state* bst, BSPLINE_Xform* bxf, 
		     int pidx, int qidx, float dc_dv[3])
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

void
report_score (char *alg, BSPLINE_Xform *bxf, 
	      Bspline_state *bst, int num_vox, double timing)
{
    int i;
    float ssd_grad_norm, ssd_grad_mean;

    /* Normalize gradient */
    ssd_grad_norm = 0;
    ssd_grad_mean = 0;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd_grad_mean += bst->ssd.grad[i];
	ssd_grad_norm += fabs (bst->ssd.grad[i]);
    }

    logfile_printf ("%s[%4d] %9.3f NV %6d GM %9.3f GN %9.3f [%9.3f secs]\n", 
		    alg, bst->it, bst->ssd.score, num_vox, ssd_grad_mean, 
		    ssd_grad_norm, timing);
}


void dump_xpm_hist (BSPLINE_MI_Hist* mi_hist, char* file_base, int iter)
{
	long i,j,k;
	int z;
	char c;

	// Graph Properties
	int graph_offset_x = 10;
	int graph_offset_y = 10;
	int graph_padding = 20;
	int graph_bar_height = 100;
	int graph_bar_width = 5;
	int graph_bar_spacing = (int)((float)graph_bar_width * (7.0/5.0));
	int graph_color_levels = 22;

	int fixed_bar_height;	// max bar height (pixels)
	int moving_bar_height;
	int joint_color;

	float fixed_scale;	// pixels per amt
	float moving_scale;
	float joint_scale;

	float moving_max_val=0;	
	float fixed_max_val=0;
	float joint_max_val=0;

	int fixed_total_width = mi_hist->fixed.bins * graph_bar_spacing;
	int moving_total_width = mi_hist->moving.bins * graph_bar_spacing;

	int graph_moving_x_pos = graph_offset_x + graph_bar_height + graph_padding;
	int graph_moving_y_pos = graph_offset_y + fixed_total_width + graph_padding + graph_bar_height;

	int graph_fixed_x_pos = graph_offset_x;
	int graph_fixed_y_pos = graph_offset_y + fixed_total_width;

	int border_padding = 5;
	int border_width = moving_total_width + 2*border_padding;
	int border_height = fixed_total_width + 2*border_padding;
	int border_x_pos = graph_moving_x_pos - border_padding;
	int border_y_pos = graph_offset_y - border_padding + (int)((float)graph_bar_width * (2.0/5.0));

	int canvas_width = 2*graph_offset_x + graph_bar_height + moving_total_width + graph_padding;
	int canvas_height = 2*graph_offset_y + graph_bar_height + fixed_total_width + graph_padding;
	
	float *m_hist = mi_hist->m_hist;
	float *f_hist = mi_hist->f_hist;
	float *j_hist = mi_hist->j_hist;
	
	// Pull out a canvas and brush!
	xpm_struct xpm;
	xpm_brush brush;

	char filename[20];

	sprintf(filename, "%s_%04i.xpm", file_base, iter);

	// ----------------------------------------------
	// Find max value for fixed
	for(i=0; i<mi_hist->fixed.bins; i++)
		if (f_hist[i] > fixed_max_val)
			fixed_max_val = f_hist[i];
	
	// Find max value for moving
	for(i=0; i<mi_hist->moving.bins; i++)
		if (m_hist[i] > moving_max_val)
			moving_max_val = m_hist[i];
	
	// Find max value for joint
	// (Ignoring bin 0)
	for(j=0; j<mi_hist->fixed.bins; j++) {
		for(i=0; i<mi_hist->moving.bins; i++) {
			if ( (i > 0) && (j > 1) )
				if (j_hist[j*mi_hist->moving.bins + i] > joint_max_val)
					joint_max_val = j_hist[j*mi_hist->moving.bins + i];
		}
	}


	// Generate scaling factors
	fixed_scale = (float)graph_bar_height / fixed_max_val;
	moving_scale = (float)graph_bar_height / moving_max_val;
	joint_scale = (float)graph_color_levels / joint_max_val;
	// ----------------------------------------------


    
	// ----------------------------------------------
	// stretch the canvas
	xpm_create (&xpm, canvas_width, canvas_height, 1);
	
	// setup the palette
	xpm_add_color (&xpm, 'a', 0xFFFFFF);	// white
	xpm_add_color (&xpm, 'b', 0x000000);	// black
	xpm_add_color (&xpm, 'z', 0xFFCC00);	// orange

	// generate a nice BLUE->RED gradient
	c = 'c';
	z = 0x0000FF;
	for (i=0; i<(graph_color_levels+1); i++)
	{
		xpm_add_color (&xpm, c, z);

		z -= 0x00000B;		// BLUE--
		z += 0x0B0000;		//  RED++

		c = (char)((int)c + 1);	// LETTER++
	}

	// Prime the XPM Canvas
	xpm_prime_canvas (&xpm, 'a');
	// ----------------------------------------------
	

	printf("Drawing Histograms... ");

	
	/* Generate Moving Histogram */
	brush.type = XPM_BOX;
	brush.color = 'b';
	brush.x_pos = graph_moving_x_pos;
	brush.y_pos = graph_moving_y_pos;
	brush.width = graph_bar_width;
	brush.height = 0;

	for(i=0; i<mi_hist->moving.bins; i++)
	{
		brush.height = (int)(m_hist[i] * moving_scale);
		brush.y_pos = graph_moving_y_pos - brush.height;
		xpm_draw(&xpm, &brush);
		brush.x_pos += graph_bar_spacing;
	}

	
	/* Generate Fixed Histogram */
	brush.type = XPM_BOX;
	brush.color = 'b';
	brush.x_pos = graph_fixed_x_pos;
	brush.y_pos = graph_fixed_y_pos;
	brush.width = 0;
	brush.height = graph_bar_width;

	for(i=0; i<mi_hist->fixed.bins; i++)
	{
		brush.width = (int)(f_hist[i] * fixed_scale);
		xpm_draw(&xpm, &brush);
		brush.y_pos -= graph_bar_spacing;
	}


	/* Generate Joint Histogram */
	brush.type = XPM_BOX;
	brush.color = 'b';
	brush.x_pos = graph_moving_x_pos;
	brush.y_pos = graph_fixed_y_pos;
	brush.width = graph_bar_width;
	brush.height = graph_bar_width;

	z = 0;
	for(j=0; j<mi_hist->fixed.bins; j++) {
		for(i=0; i<mi_hist->moving.bins; i++) {
			joint_color = (int)(j_hist[z++] * joint_scale);
			if (joint_color > 0) {
				// special handling for bin 0
				if (joint_color > graph_color_levels) {
//					printf ("Clamp @ P(%i,%i)\n", i, j);
//					brush.color = (char)(graph_color_levels + 99);
					brush.color = 'z';
				} else {
					brush.color = (char)(joint_color + 99);
				}
			} else {
				brush.color = 'a';
			}

			xpm_draw(&xpm, &brush);		
			brush.x_pos += graph_bar_spacing;
		}

		// get ready to render new row
		brush.x_pos = graph_moving_x_pos;
		brush.y_pos -= graph_bar_spacing;
	}

	/* Generate Joint Histogram Border */
	brush.type = XPM_BOX;		// top
	brush.color = 'b';
	brush.x_pos = border_x_pos;
	brush.y_pos = border_y_pos;
	brush.width = border_width;
	brush.height = 1;
	xpm_draw(&xpm, &brush);

	brush.width = 1;		// left
	brush.height = border_height;
	xpm_draw(&xpm, &brush);

	brush.width = border_width;	// bottom
	brush.height = 1;
	brush.y_pos += border_height;
	xpm_draw(&xpm, &brush);

	brush.width = 1;		// right
	brush.height = border_height;
	brush.x_pos = border_x_pos + border_width;
	brush.y_pos = border_y_pos;
	xpm_draw(&xpm, &brush);

	printf("done.\n");
	
	// Write to file
	xpm_write (&xpm, filename);
}


/* Mutual information version of implementation "C" */
static void
bspline_score_d_mi (BSPLINE_Parms *parms, 
		    Bspline_state *bst,
		    BSPLINE_Xform *bxf, 
		    Volume *fixed, 
		    Volume *moving, 
		    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
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
    Timer timer;
    double interval;
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


    plm_timer_start (&timer);

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

		/* Compute quadratic interpolation fractions */
		clamp_quadratic_interpolate_inline (mi, moving->dim[0], miqs, fxqs);
		clamp_quadratic_interpolate_inline (mj, moving->dim[1], mjqs, fyqs);
		clamp_quadratic_interpolate_inline (mk, moving->dim[2], mkqs, fzqs);

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

		// Increment voxel counter
		num_vox ++;

	    }
	}
    }

    // Dump histogram images ??
    if (parms->xpm_hist_dump)
	    dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);


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

		bspline_update_grad_b_inline (bst, bxf, pidx, qidx, dc_dv);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    interval = plm_timer_report (&timer);

    report_score ("MI", bxf, bst, num_vox, interval);
}


/* Mutual information version of implementation "C" */
static void
bspline_score_c_mi (BSPLINE_Parms *parms, 
		    Bspline_state *bst,
		    BSPLINE_Xform *bxf, 
		    Volume *fixed, 
		    Volume *moving, 
		    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
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
    Timer timer;
    double interval;
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

    plm_timer_start (&timer);

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

    // Dump histogram images ??
    if (parms->xpm_hist_dump)
	    dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);

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

		bspline_update_grad_b_inline (bst, bxf, pidx, qidx, dc_dv);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    mse_score = mse_score / num_vox;

    interval = plm_timer_report (&timer);

    report_score ("MI", bxf, bst, num_vox, interval);
}

/* Find location and index of corresponding voxel in moving image.  
   Return 1 if corresponding voxel lies within the moving image, 
   return 0 if outside the moving image.  */
static inline int
bspline_find_correspondence 
(
 float *mxyz,             /* Output: xyz coordinates in moving image (mm) */
 float *mijk,             /* Output: ijk indices in moving image (vox) */
 const float *fxyz,       /* Input:  xyz coordinates in fixed image (mm) */
 const float *dxyz,       /* Input:  displacement from fixed to moving (mm) */
 const Volume *moving     /* Input:  moving image */
 )
{
    mxyz[0] = fxyz[0] + dxyz[0];
    mijk[0] = (mxyz[0] - moving->offset[0]) / moving->pix_spacing[0];
    if (mijk[0] < -0.5 || mijk[0] > moving->dim[0] - 0.5) return 0;

    mxyz[1] = fxyz[1] + dxyz[1];
    mijk[1] = (mxyz[1] - moving->offset[1]) / moving->pix_spacing[1];
    if (mijk[1] < -0.5 || mijk[1] > moving->dim[1] - 0.5) return 0;

    mxyz[2] = fxyz[2] + dxyz[2];
    mijk[2] = (mxyz[2] - moving->offset[2]) / moving->pix_spacing[2];
    if (mijk[2] < -0.5 || mijk[2] > moving->dim[2] - 0.5) return 0;

    return 1;
}

static inline void
clamp_linear_interpolate_3d (float mijk[3], int mijk_f[3], int mijk_r[3],
			     float li_frac_1[3], float li_frac_2[3],
			     Volume *moving)
{
    clamp_linear_interpolate (mijk[0], moving->dim[0]-1, &mijk_f[0], 
			      &mijk_r[0], &li_frac_1[0], &li_frac_2[0]);
    clamp_linear_interpolate (mijk[1], moving->dim[1]-1, &mijk_f[1], 
			      &mijk_r[1], &li_frac_1[1], &li_frac_2[1]);
    clamp_linear_interpolate (mijk[2], moving->dim[2]-1, &mijk_f[2], 
			      &mijk_r[2], &li_frac_1[2], &li_frac_2[2]);
}

#define CLAMP_LINEAR_INTERPOLATE_3D(mijk, mijk_f, mijk_r, li_frac_1,	\
				    li_frac_2, moving)			\
    do {								\
	clamp_linear_interpolate (mijk[0], moving->dim[0]-1,		\
				  &mijk_f[0], &mijk_r[0],		\
				  &li_frac_1[0], &li_frac_2[0]);	\
	clamp_linear_interpolate (mijk[1], moving->dim[1]-1,		\
				  &mijk_f[1], &mijk_r[1],		\
				  &li_frac_1[1], &li_frac_2[1]);	\
	clamp_linear_interpolate (mijk[2], moving->dim[2]-1,		\
				  &mijk_f[2], &mijk_r[2],		\
				  &li_frac_1[2], &li_frac_2[2]);	\
    } while (0)


static inline float 
bspline_li_value (float fx1, float fx2, float fy1, float fy2, 
		  float fz1, float fz2, int mvf, 
		  float *m_img, Volume *moving)
{
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    float m_val;

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

    return m_val;
}

#define BSPLINE_LI_VALUE(m_val, fx1, fx2, fy1, fy2, fz1, fz2, mvf, \
			 m_img, moving)				   \
    do {							   \
	float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;		   \
	float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;		   \
								   \
	m_x1y1z1 = fx1 * fy1 * fz1 * m_img[mvf];		   \
	m_x2y1z1 = fx2 * fy1 * fz1 * m_img[mvf+1];		   \
	m_x1y2z1 = fx1 * fy2 * fz1 * m_img[mvf+moving->dim[0]];		\
	m_x2y2z1 = fx2 * fy2 * fz1 * m_img[mvf+moving->dim[0]+1];	\
	m_x1y1z2 = fx1 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]]; \
	m_x2y1z2 = fx2 * fy1 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+1]; \
	m_x1y2z2 = fx1 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]]; \
	m_x2y2z2 = fx2 * fy2 * fz2 * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1]; \
	m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1		\
		+ m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;		\
    } while (0)

#if defined (commentout)
#define BSPLINE_LI_VALUE(m_val, li_1, li_2, mvf, m_img, moving)		\
    do {								\
	float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;			\
	float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;			\
									\
	m_x1y1z1 = li_1[0] * li_1[1] * li_1[2] * m_img[mvf];		\
	m_x2y1z1 = li_2[0] * li_1[1] * li_1[2] * m_img[mvf+1];		\
	m_x1y2z1 = li_1[0] * li_2[1] * li_1[2] * m_img[mvf+moving->dim[0]]; \
	m_x2y2z1 = li_2[0] * li_2[1] * li_1[2] * m_img[mvf+moving->dim[0]+1]; \
	m_x1y1z2 = li_1[0] * li_1[1] * li_2[2] * m_img[mvf+moving->dim[1]*moving->dim[0]]; \
	m_x2y1z2 = li_2[0] * li_1[1] * li_2[2] * m_img[mvf+moving->dim[1]*moving->dim[0]+1]; \
	m_x1y2z2 = li_1[0] * li_2[1] * li_2[2] * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]]; \
	m_x2y2z2 = li_2[0] * li_2[1] * li_2[2] * m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1]; \
	m_val = m_x1y1z1 + m_x2y1z1 + m_x1y2z1 + m_x2y2z1		\
		+ m_x1y1z2 + m_x2y1z2 + m_x1y2z2 + m_x2y2z2;		\
    } while (0)
#endif



/* This only warps voxels within the ROI.  If you need the whole 
   image, call bspline_xform_extend. */
void
bspline_warp (
    Volume *vout,       /* Output image (sized and allocated) */
    Volume *vf_out,     /* Output vf (sized and allocated, can be null) */
    BSPLINE_Xform* bxf, /* Bspline transform coefficients */
    Volume *moving,     /* Input image */
    int linear_interp,  /* 1 = trilinear, 0 = nearest neighbors */
    float default_val   /* Fill in this value outside of image */
)
{
    int d;
    int vidx;
    float* vout_img = (float*) vout->img;

    int rijk[3];             /* Indices within fixed image region (vox) */
    int fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];           /* Indices within moving image (vox) */
    float fxyz[3];           /* Position within fixed image (mm) */
    float mxyz[3];           /* Position within moving image (mm) */
    int mijk_f[3], mvf;      /* Floor */
    int mijk_r[3];           /* Round */
    int p[3];
    int q[3];
    int pidx, qidx;
    float dxyz[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* m_img = (float*) moving->img;
    float m_val;

    /* A few sanity checks */
    if (vout->pix_type != PT_FLOAT) {
	fprintf (stderr, "Error: bspline_warp pix type mismatch\n");
	return;
    }
    for (d = 0; d < 3; d++) {
	if (vout->dim[d] != bxf->img_dim[d]) {
	    print_and_exit ("Error: bspline_warp dim mismatch\n");
	    return;
	}
	if (vout->offset[d] != bxf->img_origin[d]) {
	    print_and_exit ("Error: bspline_warp offset mismatch\n");
	    return;
	}
	if (vout->pix_spacing[d] != bxf->img_spacing[d]) {
	    print_and_exit ("Error: bspline_warp pix spacing mismatch\n");
	    return;
	}
    }
    if (vf_out && vf_out->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Error: bspline_warp requires interleaved vf\n");
	return;
    }

    /* Set default */
    for (vidx = 0; vidx < vout->npix; vidx++) {
	vout_img[vidx] = default_val;
    }
    if (vf_out) {
	memset (vf_out->img, 0, vf_out->pix_size * vf_out->npix);
    }
	
    for (rijk[2] = 0, fijk[2] = bxf->roi_offset[2]; rijk[2] < bxf->roi_dim[2]; rijk[2]++, fijk[2]++) {
	p[2] = rijk[2] / bxf->vox_per_rgn[2];
	q[2] = rijk[2] % bxf->vox_per_rgn[2];
	fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];
	for (rijk[1] = 0, fijk[1] = bxf->roi_offset[1]; rijk[1] < bxf->roi_dim[1]; rijk[1]++, fijk[1]++) {
	    p[1] = rijk[1] / bxf->vox_per_rgn[1];
	    q[1] = rijk[1] % bxf->vox_per_rgn[1];
	    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
	    for (rijk[0] = 0, fijk[0] = bxf->roi_offset[0]; rijk[0] < bxf->roi_dim[0]; rijk[0]++, fijk[0]++) {
		int rc;

		p[0] = rijk[0] / bxf->vox_per_rgn[0];
		q[0] = rijk[0] % bxf->vox_per_rgn[0];
		fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];

		/* Get B-spline deformation vector */
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

		/* Compute linear index of fixed image voxel */
		fv = INDEX_OF (fijk, vout->dim);

		/* Assign deformation */
		if (vf_out) {
		    float *vf_out_img = (float*) vf_out->img;
		    vf_out_img[3*fv+0] = dxyz[0];
		    vf_out_img[3*fv+1] = dxyz[1];
		    vf_out_img[3*fv+2] = dxyz[2];
		}

		/* Compute moving image coordinate of fixed image voxel */
		rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
		    dxyz, moving);

		/* If voxel is not inside moving image */
		if (!rc) continue;

		CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
		    li_1, li_2, moving);

		if (linear_interp) {
		    /* Find linear index of "corner voxel" in moving image */
		    mvf = INDEX_OF (mijk_f, moving->dim);

		    /* Compute moving image intensity using linear 
		       interpolation */
		    /* Macro is slightly faster than function */
		    BSPLINE_LI_VALUE (m_val, 
			li_1[0], li_2[0],
			li_1[1], li_2[1],
			li_1[2], li_2[2],
			mvf, m_img, moving);
		} else {
		    /* Find linear index of "nearest voxel" in moving image */
		    mvf = INDEX_OF (mijk_r, moving->dim);

		    m_val = m_img[mvf];
		}
		/* Assign warped value to output image */
		vout_img[fv] = m_val;
	    }
	}
    }
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_score_h_mse()
//
// This is a single core CPU implementation of CUDA implementation J.
// The tile "condense" method is demonstrated.
//
// See also:
//   OpenMP implementation of CUDA J: bspline_score_g_mse()
//
// AUTHOR: James A. Shackleford
// DATE: 11.22.2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_score_h_mse (
    BSPLINE_Parms *parms,
    Bspline_state *bst, 
    BSPLINE_Xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    double score_tile;
    int num_vox;

    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;

    int idx_tile;
    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    Timer timer;
    double interval;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    int idx_knot;
    int idx_set;
    int i;

    // Start timing the code
    plm_timer_start (&timer);

    // Zero out accumulators
    ssd->score = 0;
    num_vox = 0;
    score_tile = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

    // Serial across tiles
    for (idx_tile = 0; idx_tile < num_tiles; idx_tile++) {
	int rc;
	int set_num;

	int crds_tile[3];
	int crds_local[3];
	int idx_local;

	float phys_fixed[3];
	int crds_fixed[3];
	int idx_fixed;

	float dxyz[3];

	float phys_moving[3];
	float crds_moving[3];
	int crds_moving_floor[3];
	int crds_moving_round[3];
	int idx_moving_floor;
	int idx_moving_round;

	float li_1[3], li_2[3];
	float m_val, diff;
	
	float dc_dv[3];

	float sets_x[64];
	float sets_y[64];
	float sets_z[64];

	int* k_lut = (int*)malloc(64*sizeof(int));

	memset(sets_x, 0, 64*sizeof(float));
	memset(sets_y, 0, 64*sizeof(float));
	memset(sets_z, 0, 64*sizeof(float));

	// Get tile coordinates from index
	COORDS_FROM_INDEX (crds_tile, idx_tile, bxf->rdims); 

	// Serial through voxels in tile
	for (crds_local[2] = 0; crds_local[2] < bxf->vox_per_rgn[2]; crds_local[2]++) {
	    for (crds_local[1] = 0; crds_local[1] < bxf->vox_per_rgn[1]; crds_local[1]++) {
		for (crds_local[0] = 0; crds_local[0] < bxf->vox_per_rgn[0]; crds_local[0]++) {
		    float* q_lut;
					
		    // Construct coordinates into fixed image volume
		    crds_fixed[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0] * crds_tile[0] + crds_local[0];
		    crds_fixed[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1] * crds_tile[1] + crds_local[1];
		    crds_fixed[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2] * crds_tile[2] + crds_local[2];

		    // Make sure we are inside the image volume
		    if (crds_fixed[0] >= bxf->roi_offset[0] + bxf->roi_dim[0])
			continue;
		    if (crds_fixed[1] >= bxf->roi_offset[1] + bxf->roi_dim[1])
			continue;
		    if (crds_fixed[2] >= bxf->roi_offset[2] + bxf->roi_dim[2])
			continue;

		    // Compute physical coordinates of fixed image voxel
		    phys_fixed[0] = bxf->img_origin[0] + bxf->img_spacing[0] * crds_fixed[0];
		    phys_fixed[1] = bxf->img_origin[1] + bxf->img_spacing[1] * crds_fixed[1];
		    phys_fixed[2] = bxf->img_origin[2] + bxf->img_spacing[2] * crds_fixed[2];
					
		    // Construct the local index within the tile
		    idx_local = INDEX_OF (crds_local, bxf->vox_per_rgn);

		    // Construct the image volume index
		    idx_fixed = INDEX_OF (crds_fixed, fixed->dim);

		    // Calc. deformation vector (dxyz) for voxel
		    bspline_interp_pix_b_inline (dxyz, bxf, idx_tile, idx_local);

		    // Calc. moving image coordinate from the deformation vector
		    rc = bspline_find_correspondence (phys_moving,
			crds_moving,
			phys_fixed,
			dxyz,
			moving);

		    // Return code is 0 if voxel is pushed outside of moving image
		    if (!rc) continue;

		    // Compute linear interpolation fractions
		    CLAMP_LINEAR_INTERPOLATE_3D (crds_moving,
			crds_moving_floor,
			crds_moving_round,
			li_1,
			li_2,
			moving);

		    // Find linear indices for moving image
		    idx_moving_floor = INDEX_OF (crds_moving_floor, moving->dim);
		    idx_moving_round = INDEX_OF (crds_moving_round, moving->dim);

		    // Calc. moving voxel intensity via linear interpolation
		    BSPLINE_LI_VALUE (m_val, 
			li_1[0], li_2[0],
			li_1[1], li_2[1],
			li_1[2], li_2[2],
			idx_moving_floor,
			m_img, moving);

		    // Compute intensity difference
		    diff = f_img[idx_fixed] - m_val;

		    // Store the score!
		    score_tile += diff * diff;
		    num_vox++;

		    // Compute dc_dv
		    dc_dv[0] = diff * m_grad[3 * idx_moving_round + 0];
		    dc_dv[1] = diff * m_grad[3 * idx_moving_round + 1];
		    dc_dv[2] = diff * m_grad[3 * idx_moving_round + 2];
					
		    // Initialize q_lut
		    q_lut = &bxf->q_lut[64*idx_local];
					
		    // Condense dc_dv @ current voxel index
		    for (set_num = 0; set_num < 64; set_num++) {
			sets_x[set_num] += dc_dv[0] * q_lut[set_num];
			sets_y[set_num] += dc_dv[1] * q_lut[set_num];
			sets_z[set_num] += dc_dv[2] * q_lut[set_num];
		    }
		}
	    }
	}
		
	// The tile is now condensed.  Now we will put it in the
	// proper slot within the control point bin that it belong to.
					
	// Generate k_lut
	find_knots(k_lut, idx_tile, bxf->rdims, bxf->cdims);

	for (set_num = 0; set_num < 64; set_num++) {
	    int knot_num = k_lut[set_num];

	    cond_x[ (64*knot_num) + (63 - set_num) ] = sets_x[set_num];
	    cond_y[ (64*knot_num) + (63 - set_num) ] = sets_y[set_num];
	    cond_z[ (64*knot_num) + (63 - set_num) ] = sets_z[set_num];
	}

	free (k_lut);
    }

    // "Reduce"
    for (idx_knot = 0; idx_knot < (bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2]); idx_knot++) {
	for(idx_set = 0; idx_set < 64; idx_set++) {
	    ssd->grad[3*idx_knot + 0] += cond_x[64*idx_knot + idx_set];
	    ssd->grad[3*idx_knot + 1] += cond_y[64*idx_knot + idx_set];
	    ssd->grad[3*idx_knot + 2] += cond_z[64*idx_knot + idx_set];
	}
    }

    free (cond_x);
    free (cond_y);
    free (cond_z);

    ssd->score = score_tile / num_vox;

    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}


////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_score_g_mse()
//
// This is a multi-CPU implementation of CUDA implementation J.  OpenMP is
// used.  The tile "condense" method is demonstrated.
//
// AUTHOR: James A. Shackleford
// DATE: 11.22.2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_score_g_mse (
    BSPLINE_Parms *parms,
    Bspline_state *bst, 
    BSPLINE_Xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    double score_tile;
    int num_vox;

    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;

    int idx_tile;
    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    Timer timer;
    double interval;

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);

    int idx_knot;
    int idx_set;
    int i;

    // Start timing the code
    plm_timer_start (&timer);

    // Zero out accumulators
    ssd->score = 0;
    num_vox = 0;
    score_tile = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);

    // Parallel across tiles
#pragma omp parallel for reduction (+:num_vox,score_tile)
    for (idx_tile = 0; idx_tile < num_tiles; idx_tile++) {
	int rc;
	int set_num;

	int crds_tile[3];
	int crds_local[3];
	int idx_local;

	float phys_fixed[3];
	int crds_fixed[3];
	int idx_fixed;

	float dxyz[3];

	float phys_moving[3];
	float crds_moving[3];
	int crds_moving_floor[3];
	int crds_moving_round[3];
	int idx_moving_floor;
	int idx_moving_round;

	float li_1[3], li_2[3];
	float m_val, diff;
	
	float dc_dv[3];

	float sets_x[64];
	float sets_y[64];
	float sets_z[64];

	int* k_lut = (int*)malloc(64*sizeof(int));

	memset(sets_x, 0, 64*sizeof(float));
	memset(sets_y, 0, 64*sizeof(float));
	memset(sets_z, 0, 64*sizeof(float));

	// Get tile coordinates from index
	COORDS_FROM_INDEX (crds_tile, idx_tile, bxf->rdims); 

	// Serial through voxels in tile
	for (crds_local[2] = 0; crds_local[2] < bxf->vox_per_rgn[2]; crds_local[2]++) {
	    for (crds_local[1] = 0; crds_local[1] < bxf->vox_per_rgn[1]; crds_local[1]++) {
		for (crds_local[0] = 0; crds_local[0] < bxf->vox_per_rgn[0]; crds_local[0]++) {
		    float* q_lut;
					
		    // Construct coordinates into fixed image volume
		    crds_fixed[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0] * crds_tile[0] + crds_local[0];
		    crds_fixed[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1] * crds_tile[1] + crds_local[1];
		    crds_fixed[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2] * crds_tile[2] + crds_local[2];

		    // Make sure we are inside the image volume
		    if (crds_fixed[0] >= bxf->roi_offset[0] + bxf->roi_dim[0])
			continue;
		    if (crds_fixed[1] >= bxf->roi_offset[1] + bxf->roi_dim[1])
			continue;
		    if (crds_fixed[2] >= bxf->roi_offset[2] + bxf->roi_dim[2])
			continue;

		    // Compute physical coordinates of fixed image voxel
		    phys_fixed[0] = bxf->img_origin[0] + bxf->img_spacing[0] * crds_fixed[0];
		    phys_fixed[1] = bxf->img_origin[1] + bxf->img_spacing[1] * crds_fixed[1];
		    phys_fixed[2] = bxf->img_origin[2] + bxf->img_spacing[2] * crds_fixed[2];
					
		    // Construct the local index within the tile
		    idx_local = INDEX_OF (crds_local, bxf->vox_per_rgn);

		    // Construct the image volume index
		    idx_fixed = INDEX_OF (crds_fixed, fixed->dim);

		    // Calc. deformation vector (dxyz) for voxel
		    bspline_interp_pix_b_inline (dxyz, bxf, idx_tile, idx_local);

		    // Calc. moving image coordinate from the deformation vector
		    rc = bspline_find_correspondence (phys_moving,
			crds_moving,
			phys_fixed,
			dxyz,
			moving);

		    // Return code is 0 if voxel is pushed outside of moving image
		    if (!rc) continue;

		    // Compute linear interpolation fractions
		    CLAMP_LINEAR_INTERPOLATE_3D (crds_moving,
			crds_moving_floor,
			crds_moving_round,
			li_1,
			li_2,
			moving);

		    // Find linear indices for moving image
		    idx_moving_floor = INDEX_OF (crds_moving_floor, moving->dim);
		    idx_moving_round = INDEX_OF (crds_moving_round, moving->dim);

		    // Calc. moving voxel intensity via linear interpolation
		    BSPLINE_LI_VALUE (m_val, 
			li_1[0], li_2[0],
			li_1[1], li_2[1],
			li_1[2], li_2[2],
			idx_moving_floor,
			m_img, moving);

		    // Compute intensity difference
		    diff = f_img[idx_fixed] - m_val;

		    // Store the score!
		    score_tile += diff * diff;
		    num_vox++;

		    // Compute dc_dv
		    dc_dv[0] = diff * m_grad[3 * idx_moving_round + 0];
		    dc_dv[1] = diff * m_grad[3 * idx_moving_round + 1];
		    dc_dv[2] = diff * m_grad[3 * idx_moving_round + 2];
					
		    // Initialize q_lut
		    q_lut = &bxf->q_lut[64*idx_local];
					
		    // Condense dc_dv @ current voxel index
		    for (set_num = 0; set_num < 64; set_num++) {
			sets_x[set_num] += dc_dv[0] * q_lut[set_num];
			sets_y[set_num] += dc_dv[1] * q_lut[set_num];
			sets_z[set_num] += dc_dv[2] * q_lut[set_num];
		    }
		}
	    }
	}
		
	// The tile is now condensed.  Now we will put it in the
	// proper slot within the control point bin that it belong to.
					
	// Generate k_lut
	find_knots(k_lut, idx_tile, bxf->rdims, bxf->cdims);

	for (set_num = 0; set_num < 64; set_num++) {
	    int knot_num = k_lut[set_num];

	    cond_x[ (64*knot_num) + (63 - set_num) ] = sets_x[set_num];
	    cond_y[ (64*knot_num) + (63 - set_num) ] = sets_y[set_num];
	    cond_z[ (64*knot_num) + (63 - set_num) ] = sets_z[set_num];
	}

	free (k_lut);
    }

    // "Reduce"
    for (idx_knot = 0; idx_knot < (bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2]); idx_knot++) {
	for(idx_set = 0; idx_set < 64; idx_set++) {
	    ssd->grad[3*idx_knot + 0] += cond_x[64*idx_knot + idx_set];
	    ssd->grad[3*idx_knot + 1] += cond_y[64*idx_knot + idx_set];
	    ssd->grad[3*idx_knot + 2] += cond_z[64*idx_knot + idx_set];
	}
    }

    free (cond_x);
    free (cond_y);
    free (cond_z);

    ssd->score = score_tile / num_vox;

    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

/* bspline_score_f_mse:
 * This version is similar to version "D" in that it calculates the
 * score tile by tile. For the gradient, it iterates over each
 * control knot, determines which tiles influence the knot, and then
 * sums the total influence from each of those tiles. This implementation
 * allows for greater parallelization on the GPU.
 */
void bspline_score_f_mse (BSPLINE_Parms *parms,
			  Bspline_state *bst, 
			  BSPLINE_Xform *bxf,
			  Volume *fixed,
			  Volume *moving,
			  Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;

    int i;
    int qz;
    int p[3];
    float* dc_dv;
    float* f_img = (float*)fixed->img;
    float* m_img = (float*)moving->img;
    float* m_grad = (float*)moving_grad->img;
    int num_vox;
    int cidx;
    Timer timer;
    double interval;

    int total_vox_per_rgn = bxf->vox_per_rgn[0] * bxf->vox_per_rgn[1] * bxf->vox_per_rgn[2];

    //static int it = 0;
    //char debug_fn[1024];

    plm_timer_start (&timer);

    // Allocate memory for the dc_dv array. In this implementation, 
    // dc_dv values are computed for each voxel, and stored in the 
    // first phase.  Then the second phase uses the dc_dv calculation.
    dc_dv = (float*) malloc (3 * total_vox_per_rgn 
			     * bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2] 
			     * sizeof(float));

    ssd->score = 0;
    memset(ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    // Serial across tiles
    for (p[2] = 0; p[2] < bxf->rdims[2]; p[2]++) {
	for (p[1] = 0; p[1] < bxf->rdims[1]; p[1]++) {
	    for (p[0] = 0; p[0] < bxf->rdims[0]; p[0]++) {
		int tile_num_vox = 0;
		double tile_score = 0.0;
		int pidx;
		int* c_lut;

		// Compute linear index for tile
		pidx = INDEX_OF (p, bxf->rdims);

		// Find c_lut row for this tile
		c_lut = &bxf->c_lut[pidx*64];

		// Parallel across offsets within the tile
#pragma omp parallel for reduction (+:tile_num_vox,tile_score)
		for (qz = 0; qz < bxf->vox_per_rgn[2]; qz++) {
		    int q[3];
		    q[2] = qz;
		    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
			for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
			    float diff;
			    int qidx;
			    int fi, fj, fk, fv;
			    float mi, mj, mk;
			    float fx, fy, fz;
			    float mx, my, mz;
			    int mif, mjf, mkf, mvf;  /* Floor */
			    int mir, mjr, mkr, mvr;  /* Round */
			    float* dc_dv_row;
			    float dxyz[3];
			    float fx1, fx2, fy1, fy2, fz1, fz2;
			    float m_val;
			    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
			    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;

			    // Compute linear index for this offset
			    qidx = INDEX_OF (q, bxf->vox_per_rgn);

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
			    tile_score += diff * diff;
			    tile_num_vox++;

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
		ssd->score += tile_score;
		num_vox += tile_num_vox;
	    }
	}
    }

    /* Parallel across control knots */
#pragma omp parallel for
    for (cidx = 0; cidx < bxf->num_knots; cidx++) {
	int knot_x, knot_y, knot_z;
	int x_offset, y_offset, z_offset;

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
	for (z_offset = -2; z_offset < 2; z_offset++) {
	    for (y_offset = -2; y_offset < 2; y_offset++) {
		for (x_offset = -2; x_offset < 2; x_offset++) {

		    // Using the current x, y, and z offset from the control knot position,
		    // calculate the index for one of the tiles that influence this knot.
		    int tile_x, tile_y, tile_z;

		    tile_x = knot_x + x_offset;
		    tile_y = knot_y + y_offset;
		    tile_z = knot_z + z_offset;

		    // Determine if the tile lies within the volume.
		    if((tile_x >= 0 && tile_x < bxf->rdims[0]) &&
		       (tile_y >= 0 && tile_y < bxf->rdims[1]) &&
		       (tile_z >= 0 && tile_z < bxf->rdims[2])) {

			int m;
			int q[3];
			int qidx;
			int pidx;
			int* c_lut;
			float* dc_dv_row;

			// Compute linear index for tile.
			pidx = ((tile_z * bxf->rdims[1] + tile_y) * bxf->rdims[0]) + tile_x;

			// Find c_lut row for this tile.
			c_lut = &bxf->c_lut[64*pidx];

			// Pull out the dc_dv values for just this tile.
			dc_dv_row = &dc_dv[3 * total_vox_per_rgn * pidx];

			// Find the coefficient index in the c_lut row in order to determine
			// the linear index of the control point with respect to the current tile.
			for (m = 0; m < 64; m++) {
			    if (c_lut[m] == cidx) {
				break;
			    }
			}

			// Iterate through each of the offsets within the tile and 
			// accumulate the gradient.
			for (qidx = 0, q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
			    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
				for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++, qidx++) {

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

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

void
bspline_score_e_mse (BSPLINE_Parms *parms, 
		     Bspline_state *bst,
		     BSPLINE_Xform* bxf, 
		     Volume *fixed, 
		     Volume *moving, 
		     Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int p[3];
    int s[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    int num_vox;
    Timer timer;
    double interval;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    /* Serial across 64 groups of tiles */
    for (s[2] = 0; s[2] < 4; s[2]++) {
	for (s[1] = 0; s[1] < 4; s[1]++) {
	    for (s[0] = 0; s[0] < 4; s[0]++) {
		int tile;
		int tilelist_len = 0;
		int *tilelist = 0;
		int tile_num_vox = 0;
		double tile_score = 0.0;

		/* Create list of tiles in the group */
		for (p[2] = s[2]; p[2] < bxf->rdims[2]; p[2]+=4) {
		    for (p[1] = s[1]; p[1] < bxf->rdims[1]; p[1]+=4) {
			for (p[0] = s[0]; p[0] < bxf->rdims[0]; p[0]+=4) {
			    tilelist_len ++;
			    tilelist = realloc (tilelist, 3 * tilelist_len * sizeof(int));
			    tilelist[3*tilelist_len - 3] = p[0];
			    tilelist[3*tilelist_len - 2] = p[1];
			    tilelist[3*tilelist_len - 1] = p[2];
			}
		    }
		}

		//		printf ("%d tiles in parallel...\n");

		/* Parallel across tiles within groups */
#pragma omp parallel for reduction (+:tile_num_vox,tile_score)
		for (tile = 0; tile < tilelist_len; tile++) {
		    int pidx;
		    int* c_lut;
		    int p[3];
		    int q[3];

		    p[0] = tilelist[3*tile + 0];
		    p[1] = tilelist[3*tile + 1];
		    p[2] = tilelist[3*tile + 2];

		    /* Compute linear index for tile */
		    pidx = INDEX_OF (p, bxf->rdims);

		    /* Find c_lut row for this tile */
		    c_lut = &bxf->c_lut[pidx*64];

		    //logfile_printf ("Kernel 1, tile %d %d %d\n", p[0], p[1], p[2]);

		    /* Serial across offsets */
		    for (q[2] = 0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
			for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
			    for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
				int qidx;
				int fi, fj, fk, fv;
				float mx, my, mz;
				float mi, mj, mk;
				float fx, fy, fz;
				int mif, mjf, mkf, mvf;  /* Floor */
				int mir, mjr, mkr, mvr;  /* Round */
				float fx1, fx2, fy1, fy2, fz1, fz2;
				float dxyz[3];
				float m_val;
				float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
				float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
				float diff;
				float dc_dv[3];
				int m;

				/* Compute linear index for this offset */
				qidx = INDEX_OF (q, bxf->vox_per_rgn);

				/* Get (i,j,k) index of the voxel */
				fi = bxf->roi_offset[0] + p[0] * bxf->vox_per_rgn[0] + q[0];
				fj = bxf->roi_offset[1] + p[1] * bxf->vox_per_rgn[1] + q[1];
				fk = bxf->roi_offset[2] + p[2] * bxf->vox_per_rgn[2] + q[2];

				/* Some of the pixels are outside image */
				if (fi >= bxf->roi_offset[0] + bxf->roi_dim[0]) continue;
				if (fj >= bxf->roi_offset[1] + bxf->roi_dim[1]) continue;
				if (fk >= bxf->roi_offset[2] + bxf->roi_dim[2]) continue;

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
				tile_score += diff * diff;
				tile_num_vox ++;

				/* Compute spatial gradient using nearest neighbors */
				mvr = (mkr * moving->dim[1] + mjr) * moving->dim[0] + mir;

				/* Store dc_dv for this offset */
				dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
				dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
				dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */

				/* Serial across 64 control points */
				for (m = 0; m < 64; m++) {
				    float* q_lut;
				    int cidx;

				    /* Find index of control point within coefficient array */
				    cidx = c_lut[m] * 3;

				    /* Find q_lut row for this offset */
				    q_lut = &bxf->q_lut[qidx*64];

				    /* Accumulate update to gradient for this 
				       control point */
				    ssd->grad[cidx+0] += dc_dv[0] * q_lut[m];
				    ssd->grad[cidx+1] += dc_dv[1] * q_lut[m];
				    ssd->grad[cidx+2] += dc_dv[2] * q_lut[m];
				}
			    }
			}
		    }
		}

		num_vox += tile_num_vox;
		ssd->score += tile_score;

		free (tilelist);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
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
bspline_score_d_mse (BSPLINE_Parms *parms, 
		     Bspline_state *bst,
		     BSPLINE_Xform* bxf, 
		     Volume *fixed, 
		     Volume *moving, 
		     Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int qz;
    int p[3];
    float* dc_dv;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    int num_vox;
    Timer timer;
    double interval;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    dc_dv = (float*) malloc (3*bxf->vox_per_rgn[0]*bxf->vox_per_rgn[1]*bxf->vox_per_rgn[2]*sizeof(float));
    ssd->score = 0;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;

    /* Serial across tiles */
    for (p[2] = 0; p[2] < bxf->rdims[2]; p[2]++) {
	for (p[1] = 0; p[1] < bxf->rdims[1]; p[1]++) {
	    for (p[0] = 0; p[0] < bxf->rdims[0]; p[0]++) {
		int k;
		int pidx;
		int* c_lut;
		int tile_num_vox = 0;
		double tile_score = 0.0;

		/* Compute linear index for tile */
		pidx = INDEX_OF (p, bxf->rdims);

		/* Find c_lut row for this tile */
		c_lut = &bxf->c_lut[pidx*64];

		//logfile_printf ("Kernel 1, tile %d %d %d\n", p[0], p[1], p[2]);

		/* Parallel across offsets */
#pragma omp parallel for reduction (+:tile_num_vox,tile_score)
		for (qz = 0; qz < bxf->vox_per_rgn[2]; qz++) {
		    int q[3];
		    q[2] = qz;
		    for (q[1] = 0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
			for (q[0] = 0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
			    int qidx;
			    int fi, fj, fk, fv;
			    float mx, my, mz;
			    float mi, mj, mk;
			    float fx, fy, fz;
			    int mif, mjf, mkf, mvf;  /* Floor */
			    int mir, mjr, mkr, mvr;  /* Round */
			    float fx1, fx2, fy1, fy2, fz1, fz2;
			    float dxyz[3];
			    float m_val;
			    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
			    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
			    float diff;

			    /* Compute linear index for this offset */
			    qidx = INDEX_OF (q, bxf->vox_per_rgn);

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
			    tile_score += diff * diff;
			    tile_num_vox ++;

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
		num_vox += tile_num_vox;
		ssd->score += tile_score;

		/* Parallel across 64 control points */
#pragma omp parallel for
		for (k = 0; k < 4; k++) {
		    int i, j;
		    for (j = 0; j < 4; j++) {
			for (i = 0; i < 4; i++) {
			    int qidx;
			    int cidx;
			    int q[3];
			    int m;

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

    /* Normalize score for MSE */
    ssd->score = ssd->score / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

/* Mean-squared error version of implementation "C" */
/* ----- This is the best known version for single processor CPU's ----- */
/* Implementation "C" is slower than "B", but yields a smoother cost function 
   for use by L-BFGS-B.  It uses linear interpolation of moving image, 
   and nearest neighbor interpolation of gradient */
void
bspline_score_c_mse (
    BSPLINE_Parms *parms, 
    Bspline_state *bst,
    BSPLINE_Xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
)
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    int rijk[3];             /* Indices within fixed image region (vox) */
    int fijk[3], fv;         /* Indices within fixed image (vox) */
    float mijk[3];           /* Indices within moving image (vox) */
    float fxyz[3];           /* Position within fixed image (mm) */
    float mxyz[3];           /* Position within moving image (mm) */
    int mijk_f[3], mvf;      /* Floor */
    int mijk_r[3], mvr;      /* Round */
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* m_grad = (float*) moving_grad->img;
    float dxyz[3];
    int num_vox;
    int pidx, qidx;
    Timer timer;
    double interval;
    float m_val;

    /* GCS: Oct 5, 2009.  We have determined that sequential accumulation
       of the score requires double precision.  However, reduction 
       accumulation does not. */
    double score_acc = 0.;

    static int it = 0;
    char debug_fn[1024];
    FILE* fp;

    if (parms->debug) {
	sprintf (debug_fn, "dc_dv_mse_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    ssd->score = 0.0f;
    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    num_vox = 0;
    for (rijk[2] = 0, fijk[2] = bxf->roi_offset[2]; rijk[2] < bxf->roi_dim[2]; rijk[2]++, fijk[2]++) {
	p[2] = rijk[2] / bxf->vox_per_rgn[2];
	q[2] = rijk[2] % bxf->vox_per_rgn[2];
	fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];
	for (rijk[1] = 0, fijk[1] = bxf->roi_offset[1]; rijk[1] < bxf->roi_dim[1]; rijk[1]++, fijk[1]++) {
	    p[1] = rijk[1] / bxf->vox_per_rgn[1];
	    q[1] = rijk[1] % bxf->vox_per_rgn[1];
	    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
	    for (rijk[0] = 0, fijk[0] = bxf->roi_offset[0]; rijk[0] < bxf->roi_dim[0]; rijk[0]++, fijk[0]++) {
		int rc;
		p[0] = rijk[0] / bxf->vox_per_rgn[0];
		q[0] = rijk[0] % bxf->vox_per_rgn[0];
		fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];

		/* Get B-spline deformation vector */
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

		/* Compute moving image coordinate of fixed image voxel */
		rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
						  dxyz, moving);

		/* If voxel is not inside moving image */
		if (!rc) continue;

		/* Compute interpolation fractions */
		CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
					     li_1, li_2, moving);

		/* Find linear index of "corner voxel" in moving image */
		mvf = INDEX_OF (mijk_f, moving->dim);

		/* Compute moving image intensity using linear interpolation */
		/* Macro is slightly faster than function */
		BSPLINE_LI_VALUE (m_val, 
				  li_1[0], li_2[0],
				  li_1[1], li_2[1],
				  li_1[2], li_2[2],
				  mvf, m_img, moving);

		/* Compute linear index of fixed image voxel */
		fv = INDEX_OF (fijk, fixed->dim);

		/* Compute intensity difference */
		diff = f_img[fv] - m_val;

		/* Compute spatial gradient using nearest neighbors */
		mvr = INDEX_OF (mijk_r, moving->dim);
		dc_dv[0] = diff * m_grad[3*mvr+0];  /* x component */
		dc_dv[1] = diff * m_grad[3*mvr+1];  /* y component */
		dc_dv[2] = diff * m_grad[3*mvr+2];  /* z component */
		bspline_update_grad_b_inline (bst, bxf, pidx, qidx, dc_dv);
		
		if (parms->debug) {
		    fprintf (fp, "%d %d %d %g %g %g\n", 
			     rijk[0], rijk[1], rijk[2], 
			     dc_dv[0], dc_dv[1], dc_dv[2]);
		}
		score_acc += diff * diff;
		num_vox ++;
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    /* Normalize score for MSE */
    ssd->score = score_acc / num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] = 2 * ssd->grad[i] / num_vox;
    }

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

/* This is the fastest known version.  It does nearest neighbors 
   interpolation of both moving image and gradient which doesn't 
   work with stock L-BFGS-B optimizer. */
void
bspline_score_b_mse 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst,
 BSPLINE_Xform *bxf, 
 Volume *fixed, 
 Volume *moving, 
 Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
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
    Timer timer;
    double interval;

    plm_timer_start (&timer);

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
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
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

		bspline_update_grad_b (bst, bxf, pidx, qidx, dc_dv);
		
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

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

void
bspline_score_a_mse 
(
 BSPLINE_Parms *parms, 
 Bspline_state *bst,
 BSPLINE_Xform* bxf, 
 Volume *fixed, 
 Volume *moving, 
 Volume *moving_grad
 )
{
    BSPLINE_Score* ssd = &bst->ssd;
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
    Timer timer;
    double interval;

    plm_timer_start (&timer);

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
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
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
		bspline_update_grad (bst, bxf, p, qidx, dc_dv);
		
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

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

void
bspline_score (BSPLINE_Parms *parms, 
	       Bspline_state *bst,
	       BSPLINE_Xform* bxf, 
	       Volume *fixed, 
	       Volume *moving, 
	       Volume *moving_grad)
{
#if (CUDA_FOUND)
    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {
	switch (parms->implementation) {
	case 'c':
	    bspline_cuda_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'd':
	    bspline_cuda_score_d_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'e':
	    bspline_cuda_score_e_mse_v2 (parms, bst, bxf, fixed, moving, moving_grad);
	    //bspline_cuda_score_e_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'f':
	    bspline_cuda_score_f_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'g':
	    bspline_cuda_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'h':
	    bspline_cuda_score_h_mse (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	case 'i':
	    bspline_cuda_score_i_mse (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	default:
	    bspline_cuda_score_j_mse (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	}
	return;
    } else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {
	switch (parms->implementation) {
	case 'a':
	    bspline_cuda_MI_a (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	default:
	    bspline_cuda_MI_a (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	}

    }
#endif

    if (parms->metric == BMET_MSE) {
	switch (parms->implementation) {
	case 'a':
	    bspline_score_a_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'b':
	    bspline_score_b_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'c':
	    bspline_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'd':
	    bspline_score_d_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'e':
	    bspline_score_e_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'f':
	    bspline_score_f_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'g':
	    bspline_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'h':
	    bspline_score_h_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	}
    }

    if (parms->metric == BMET_MI) {
	switch (parms->implementation) {
	case 'c':
	    bspline_score_c_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'd':
	    bspline_score_d_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_score_c_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	}
    }
}

void
bspline_optimize_steepest (
			   BSPLINE_Xform *bxf, 
			   Bspline_state *bst, 
			   BSPLINE_Parms *parms, 
			   Volume *fixed, 
			   Volume *moving, 
			   Volume *moving_grad
			   )
{
    BSPLINE_Score* ssd = &bst->ssd;
    int i;
    //    float a = 0.003f;
    //    float alpha = 0.5f, A = 10.0f;
    float a, gamma;
    float gain = 1.5;
    float ssd_grad_norm;
    float old_score;
    FILE* fp;

    if (parms->debug) {
	fp = fopen("scores.txt", "w");
    }

    /* Set iteration */
    bst->it = 0;

    /* Get score and gradient */
    bspline_score (parms, bst, bxf, fixed, moving, moving_grad);
    old_score = bst->ssd.score;

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
    /* Save some debugging information */
    bspline_save_debug_state (parms, bst, bxf);
    if (parms->debug) {
	fprintf (fp, "%f\n", ssd->score);
    }

    while (bst->it < parms->max_its) {

	/* Update iteration number */
	bst->it ++;

	logfile_printf ("Beginning iteration %d, gamma = %g\n", bst->it, gamma);

	/* Update b-spline coefficients from gradient */
	//gamma = a / pow(it + A, alpha);
	for (i = 0; i < bxf->num_coeff; i++) {
	    bxf->coeff[i] = bxf->coeff[i] + gamma * ssd->grad[i];
	}

	/* Get score and gradient */
	bspline_score (parms, bst, bxf, fixed, moving, moving_grad);

	/* Update gamma */
	if (bst->ssd.score < old_score) {
	    gamma *= gain;
	} else {
	    gamma /= gain;
	}
	old_score = bst->ssd.score;

	/* Give a little feedback to the user */
	bspline_display_coeff_stats (bxf);
	/* Save some debugging information */
	bspline_save_debug_state (parms, bst, bxf);
	if (parms->debug) {
	    fprintf (fp, "%f\n", ssd->score);
	}
    }

    if (parms->debug) {
	fclose (fp);
    }
}

void
bspline_optimize (BSPLINE_Xform* bxf, 
		  Bspline_state **bst_in, 
		  BSPLINE_Parms *parms, 
		  Volume *fixed, 
		  Volume *moving, 
		  Volume *moving_grad)
{
    Bspline_state *bst;

#if (CUDA_FOUND)
    Dev_Pointers_Bspline dev_mem;
    Dev_Pointers_Bspline* dev_ptrs = &dev_mem;
#endif

    bst = bspline_state_create (bxf);
    log_parms (parms);
    log_bxf_header (bxf);

#if (CUDA_FOUND)
    bst->dev_ptrs = dev_ptrs;
    if( (parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE) ) {
	switch (parms->implementation) {
	case 'c':
	    bspline_cuda_initialize (fixed, moving, moving_grad, bxf, parms);
	    /* fall through */
	case 'd':
	    bspline_cuda_initialize_d (fixed, moving, moving_grad, bxf, parms);
	    break;
	case 'e':
	    bspline_cuda_initialize_e_v2 (fixed, moving, moving_grad, 
					  bxf, parms);
	    // bspline_cuda_initialize_e (fixed, moving, moving_grad, bxf, parms);
	    break;
	case 'f':
	    bspline_cuda_initialize_f (fixed, moving, moving_grad, bxf, parms);
	    break;
	case 'g':
	    bspline_cuda_initialize_g (fixed, moving, moving_grad, bxf, parms);
	    break;
	case 'h':
	    bspline_cuda_initialize_h (dev_ptrs, fixed, moving, 
				       moving_grad, bxf, parms);
	    break;
	case 'i':
	    // i now uses j's init and cleanup routines
	    bspline_cuda_initialize_j (dev_ptrs, fixed, moving, 
				       moving_grad, bxf, parms);
	    break;
	case 'j':
	case '\0':   /* Default */
	    bspline_cuda_initialize_j (dev_ptrs, fixed, moving, 
				       moving_grad, bxf, parms);
	    break;
	default:
	    printf ("Warning: option -f %c unavailble.  Switching to -f j\n",
		    parms->implementation);
	    bspline_cuda_initialize_j (dev_ptrs, fixed, moving, 
				       moving_grad, bxf, parms);
	    break;
	}
    } else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {
	switch (parms->implementation) {
	case 'a':
		bspline_cuda_init_MI_a (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
		break;
	default:
		printf ("Warning: option -f %c unavailble.  Defaulting to -f a\n", parms->implementation);
		bspline_cuda_init_MI_a (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
		break;
	}

    }
#endif

    if (parms->metric == BMET_MI) {
	bspline_initialize_mi (parms, fixed, moving);
    }

    if (parms->optimization == BOPT_LBFGSB) {
#if (FORTRAN_FOUND)
	bspline_optimize_lbfgsb (bxf, bst, parms, fixed, moving, moving_grad);
#else
	logfile_printf (
	    "LBFGSB not compiled for this platform (no fortran compiler, "
	    "no f2c library).\n  Reverting to steepest descent.\n"
	    );
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
#endif
    } else {
	bspline_optimize_steepest (bxf, bst, parms, fixed, moving, moving_grad);
    }

#if (CUDA_FOUND)
    if (parms->threading == BTHR_CUDA) {
	switch (parms->implementation) {
	case 'c':
	    bspline_cuda_clean_up ();
	    /* fall through */
	case 'd':
	case 'e':
	    bspline_cuda_clean_up_d (); // Handles versions D and E
	    break;
	case 'f':
	    bspline_cuda_clean_up_f ();
	    break;
	case 'g':
	    bspline_cuda_clean_up_g ();
	    break;
	case 'h':
	    bspline_cuda_clean_up_h (dev_ptrs);
	    break;
	case 'i':
	    // i now uses j's init and cleanup routines
	    bspline_cuda_clean_up_j (dev_ptrs);
	    break;
	default:
	    bspline_cuda_clean_up_j (dev_ptrs);
	}
    }
#endif

    if (bst_in) {
	*bst_in = bst;
    } else {
	bspline_state_free (bst);
    }
}
