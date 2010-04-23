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

    qlut = Multiplier LUT
    clut = Index LUT
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
#include "bspline_gradient.h"
#include "bspline_landmarks.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "plm_path.h"
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
   Initialization and teardown
   ----------------------------------------------------------------------- */
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
    parms->lbfgsb_factr = 1.0e+7;
    parms->lbfgsb_pgtol = 1.0e-5;
    
	parms->landmarks = 0;
    parms->landmark_stiffness = 1.0;
	parms->landmark_implementation = 'a';
	parms->young_modulus = 0.0;
	
	parms->rbf_radius = 0.0;

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

static void
bspline_cuda_state_create (
    Bspline_state *bst,           /* Modified in routine */
    BSPLINE_Xform* bxf,
    BSPLINE_Parms *parms,
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
#if (CUDA_FOUND)
    Dev_Pointers_Bspline* dev_ptrs 
	= (Dev_Pointers_Bspline*) malloc (sizeof (Dev_Pointers_Bspline));

    bst->dev_ptrs = dev_ptrs;
    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {
	switch (parms->implementation) {
	case 'i':
	case 'j':
	case '\0':   /* Default */
	    /* i and j use the same init and cleanup routines */
	    bspline_cuda_initialize_j (dev_ptrs, fixed, moving, moving_grad, 
		bxf, parms);
	break;
	default:
	    printf ("Warning: option -f %c unavailble.  Switching to -f j\n", 
		parms->implementation);
	    bspline_cuda_initialize_j (dev_ptrs, fixed, moving, moving_grad, 
		bxf, parms);
	    break;
	}
    } 
    else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {
	switch (parms->implementation) {
	case 'a':
	    bspline_cuda_init_MI_a (dev_ptrs, fixed, moving, moving_grad, 
		bxf, parms);
	    break;
	default:
	    printf ("Warning: option -f %c unavailble.  Defaulting to -f a\n",
		parms->implementation);
	    bspline_cuda_init_MI_a (dev_ptrs, fixed, moving, moving_grad, 
		bxf, parms);
	    break;
	}

    }
    else {
	printf ("No cuda initialization performed.\n");
    }
#endif
}

Bspline_state *
bspline_state_create (
    BSPLINE_Xform *bxf, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_state *bst = (Bspline_state*) malloc (sizeof (Bspline_state));
    memset (bst, 0, sizeof (Bspline_state));
    bst->ssd.grad = (float*) malloc (bxf->num_coeff * sizeof(float));
    memset (bst->ssd.grad, 0, bxf->num_coeff * sizeof(float));

    bspline_cuda_state_create (bst, bxf, parms, fixed, moving, moving_grad);

    return bst;
}

void
write_bxf (char* filename, BSPLINE_Xform* bxf)
{
    FILE* fp;
	
    fp = fopen (filename, "wb");
    if (!fp) return;

    fprintf (fp, "MGH_GPUIT_BSP <experimental>\n");
    fprintf (fp, "img_origin = %f %f %f\n", 
	bxf->img_origin[0], bxf->img_origin[1], bxf->img_origin[2]);
    fprintf (fp, "img_spacing = %f %f %f\n", 
	bxf->img_spacing[0], bxf->img_spacing[1], bxf->img_spacing[2]);
    fprintf (fp, "img_dim = %d %d %d\n", 
	bxf->img_dim[0], bxf->img_dim[1], bxf->img_dim[2]);
    fprintf (fp, "roi_offset = %d %d %d\n", 
	bxf->roi_offset[0], bxf->roi_offset[1], bxf->roi_offset[2]);
    fprintf (fp, "roi_dim = %d %d %d\n", 
	bxf->roi_dim[0], bxf->roi_dim[1], bxf->roi_dim[2]);
    fprintf (fp, "vox_per_rgn = %d %d %d\n", 
	bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
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
dump_hist (BSPLINE_MI_Hist* mi_hist, int it)
{
    float *f_hist = mi_hist->f_hist;
    float *m_hist = mi_hist->m_hist;
    float *j_hist = mi_hist->j_hist;
    int i, j, v;
    FILE *fp;
    char fn[_MAX_PATH];

    sprintf (fn, "hist_fix_%02d.csv", it);
    fp = fopen (fn, "wb");
    if (!fp) return;
    for (i = 0; i < mi_hist->fixed.bins; i++) {
	fprintf (fp, "%d %f\n", i, f_hist[i]);
    }
    fclose (fp);

    sprintf (fn, "hist_mov_%02d.csv", it);
    fp = fopen (fn, "wb");
    if (!fp) return;
    for (i = 0; i < mi_hist->moving.bins; i++) {
	fprintf (fp, "%d %f\n", i, m_hist[i]);
    }
    fclose (fp);

    sprintf (fn, "hist_jnt_%02d.csv", it);
    fp = fopen (fn, "wb");
    if (!fp) return;
    for (i = 0, v = 0; i < mi_hist->fixed.bins; i++) {
	for (j = 0; j < mi_hist->moving.bins; j++, v++) {
	    if (j_hist[v] > 0) {
		fprintf (fp, "%d %d %d %g\n", i, j, v, j_hist[v]);
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
	    dump_hist (&parms->mi_hist, bst->it);
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

void
bspline_xform_initialize 
(
 BSPLINE_Xform* bxf,         /* Output: bxf is initialized */
 float img_origin[3],        /* Image origin (in mm) */
 float img_spacing[3],       /* Image spacing (in mm) */
 int img_dim[3],             /* Image size (in vox) */
 int roi_offset[3],          /* Position of first vox in ROI (in vox) */
 int roi_dim[3],             /* Dimension of ROI (in vox) */
 int vox_per_rgn[3])         /* Knot spacing (in vox) */
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

/* -----------------------------------------------------------------------
   This extends the bspline grid.  Note, that the new roi_offset 
    in the bxf will not be the same as the one requested, because 
    bxf routines implicitly require that the first voxel of the 
    ROI matches the position of the control point. 
   ----------------------------------------------------------------------- */
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

void
bspline_initialize_mi (BSPLINE_Parms* parms, Volume* fixed, Volume* moving)
{
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    mi_hist->m_hist = (float*) malloc (sizeof (float) * mi_hist->moving.bins);
    mi_hist->f_hist = (float*) malloc (sizeof (float) * mi_hist->fixed.bins);
    mi_hist->j_hist = (float*) malloc (sizeof (float) * mi_hist->fixed.bins * mi_hist->moving.bins);
#ifdef DOUBLE_HISTS
    mi_hist->m_hist_d = (double*) malloc (sizeof (double) * mi_hist->moving.bins);
    mi_hist->f_hist_d = (double*) malloc (sizeof (double) * mi_hist->fixed.bins);
    mi_hist->j_hist_d = (double*) malloc (sizeof (double) * mi_hist->fixed.bins * mi_hist->moving.bins);
#endif
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
bspline_state_destroy (Bspline_state* bst)
{
    if (bst->ssd.grad) {
	free (bst->ssd.grad);
    }

#if (CUDA_FOUND)
    /* Both 'i', and 'j' use this routine */
    bspline_cuda_clean_up_j (bst->dev_ptrs);
#endif

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
#ifdef DOUBLE_HISTS
    double* f_hist = mi_hist->f_hist_d;
    double* m_hist = mi_hist->m_hist_d;
    double* j_hist = mi_hist->j_hist_d;
#else
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
#endif
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];

    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, mi_hist, 
	f_val, m_val);

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
    double fnv = (double) num_vox;
    double score = 0;
    double hist_thresh = 0.001 / (mi_hist->moving.bins * mi_hist->fixed.bins);

    /* Compute cost */
    for (i = 0, v = 0; i < mi_hist->fixed.bins; i++) {
	for (j = 0; j < mi_hist->moving.bins; j++, v++) {
	    if (j_hist[v] > hist_thresh) {
		score -= j_hist[v] * logf (fnv * j_hist[v] / (m_hist[j] * f_hist[i]));
	    }
	}
    }

    score = score / fnv;
    return (float) score;
}

void
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
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		v = (k+bxf->roi_offset[2]) * interp->dim[0] * interp->dim[1]
		    + (j+bxf->roi_offset[1]) * interp->dim[0] 
		    + (i+bxf->roi_offset[0]);
		out = &img[3*v];
		bspline_interp_pix (out, bxf, p, qidx);
	    }
	}
    }
}

void
bspline_update_grad (
    Bspline_state *bst, 
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
    float ma,           /* Input: (Unrounded) pixel coordinate (in vox units) */
    long dmax,          /* Input: Maximum coordinate in this dimension */
    long maqs[3],       /* Output: x, y, or z coord of 3 pixels in moving img */
    float faqs[3]       /* Output: Fraction of interpolant for 3 voxels */
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
    float ma,          /* Input: (Unrounded) pixel coordinate (in vox units) */
    long dmax,         /* Input: Maximum coordinate in this dimension */
    long maqs[3],      /* Output: x, y, or z coord of 3 pixels in moving img */
    float faqs[3]      /* Output: Gradient interpolant for 3 voxels */
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

    // JAS 04.19.2010
    // MI scores are between 0 and 1
    // The extra decimal point resolution helps in seeing
    // if the optimizer is performing adaquately.
    if (alg == "MI")
    {
	    logfile_printf ("%s[%4d] %1.6f NV %6d GM %9.3f GN %9.3f [%9.3f secs]\n", 
		    alg, bst->it, bst->ssd.score, num_vox, ssd_grad_mean, 
		    ssd_grad_norm, timing);
    } else {
	    logfile_printf ("%s[%4d] %9.3f NV %6d GM %9.3f GN %9.3f [%9.3f secs]\n", 
		    alg, bst->it, bst->ssd.score, num_vox, ssd_grad_mean, 
		    ssd_grad_norm, timing);
    }
}

void dump_xpm_hist (BSPLINE_MI_Hist* mi_hist, char* file_base, int iter)
{
    long i,j;
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

    //	int fixed_bar_height;	// max bar height (pixels)
    //	int moving_bar_height;
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
		    //	printf ("Clamp @ P(%i,%i)\n", i, j);
		    //	brush.color = (char)(graph_color_levels + 99);
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

/* -----------------------------------------------------------------------
   Macros
   ----------------------------------------------------------------------- */
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

static inline void
bspline_mi_hist_add_pvi_8 (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float li_1[3],           /* Fraction of interpolant in lower index */
    float li_2[3])           /* Fraction of interpolant in upper index */
{
    float m_x1y1z1, m_x2y1z1, m_x1y2z1, m_x2y2z1;
    float m_x1y1z2, m_x2y1z2, m_x1y2z2, m_x2y2z2;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;

    m_x1y1z1 = li_1[0] * li_1[1] * li_1[2];
    m_x2y1z1 = li_2[0] * li_1[1] * li_1[2];
    m_x1y2z1 = li_1[0] * li_2[1] * li_1[2];
    m_x2y2z1 = li_2[0] * li_2[1] * li_1[2];
    m_x1y1z2 = li_1[0] * li_1[1] * li_2[2];
    m_x2y1z2 = li_2[0] * li_1[1] * li_2[2];
    m_x1y2z2 = li_1[0] * li_2[1] * li_2[2];
    m_x2y2z2 = li_2[0] * li_2[1] * li_2[2];

    /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf], m_x1y1z1);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+1], m_x2y1z1);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[0]], m_x1y2z1);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[0]+1], m_x2y2z1);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]], m_x1y1z2);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+1], m_x2y1z2);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]], m_x1y2z2);
    bspline_mi_hist_add (mi_hist, f_img[fv], m_img[mvf+moving->dim[1]*moving->dim[0]+moving->dim[0]+1], m_x2y2z2);
}

static inline void
bspline_mi_hist_add_pvi_6 (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float mijk[3]
)
{
    long miqs[3], mjqs[3], mkqs[3];	/* Rounded indices */
    float fxqs[3], fyqs[3], fzqs[3];	/* Fractional values */
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    const float ONE_THIRD = 1.0f / 3.0f;

    /* Compute quadratic interpolation fractions */
    clamp_quadratic_interpolate_inline (mijk[0], moving->dim[0], miqs, fxqs);
    clamp_quadratic_interpolate_inline (mijk[1], moving->dim[1], mjqs, fyqs);
    clamp_quadratic_interpolate_inline (mijk[2], moving->dim[2], mkqs, fzqs);

#if 0
    printf ("[%d %d %d], [%d %d %d], [%d %d %d]\n",
	miqs[0], miqs[1], miqs[2],
	mjqs[0], mjqs[1], mjqs[2],
	mkqs[0], mkqs[1], mkqs[2]
    );
    printf ("[%f %f %f], [%f %f %f], [%f %f %f]\n",
	fxqs[0], fxqs[1], fxqs[2], 
	fyqs[0], fyqs[1], fyqs[2], 
	fzqs[0], fzqs[1], fzqs[2]
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
}

/* -----------------------------------------------------------------------
   bspline_mi_pvi_8_dc_dv
   bspline_mi_pvi_6_dc_dv

   Compute pixel contribution to gradient based on histogram change

   There are 6 or 8 correspondences between fixed and moving.  
   Each of these correspondences will update 2 or 3 histogram bins
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
   ----------------------------------------------------------------------- */
static inline void
bspline_mi_pvi_8_dc_dv (
    float dc_dv[3],                /* Output */
    BSPLINE_MI_Hist* mi_hist,      /* Input */
    Bspline_state *bst,            /* Input */
    Volume *fixed,                 /* Input */
    Volume *moving,                /* Input */
    int fv,                        /* Input */
    int mvf,                       /* Input */
    float mijk[3],                 /* Input */
    float num_vox_f                /* Input */
)
{
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];
    float dS_dP;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
    BSPLINE_Score* ssd = &bst->ssd;
    int debug = 0;

    dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

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
}

static inline void
bspline_mi_pvi_6_dc_dv (
    float dc_dv[3],                /* Output */
    BSPLINE_MI_Hist* mi_hist,      /* Input */
    Bspline_state *bst,            /* Input */
    Volume *fixed,                 /* Input */
    Volume *moving,                /* Input */
    int fv,                        /* Input */
    int mvf,                       /* Input */
    float mijk[3],                 /* Input */
    float num_vox_f                /* Input */
)
{
    long miqs[3], mjqs[3], mkqs[3];	/* Rounded indices */
    float fxqs[3], fyqs[3], fzqs[3];	/* Fractional values */
    long j_idxs[2];
    long m_idxs[2];
    long f_idxs[1];
    float fxs[2];
    float dS_dP;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
    BSPLINE_Score* ssd = &bst->ssd;
    int debug = 0;

    dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

    /* Compute quadratic interpolation fractions */
    clamp_quadratic_interpolate_grad_inline (mijk[0], moving->dim[0], 
	miqs, fxqs);
    clamp_quadratic_interpolate_grad_inline (mijk[1], moving->dim[1], 
	mjqs, fyqs);
    clamp_quadratic_interpolate_grad_inline (mijk[2], moving->dim[2], 
	mkqs, fzqs);

    /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[0] = - fxqs[1] * dS_dP;
    dc_dv[1] = - fyqs[1] * dS_dP;
    dc_dv[2] = - fzqs[1] * dS_dP;
#else
    dc_dv[0] -= - fxqs[1] * dS_dP;
    dc_dv[1] -= - fyqs[1] * dS_dP;
    dc_dv[2] -= - fzqs[1] * dS_dP;
#endif

    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[0];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[0] = - fxqs[0] * dS_dP;
#else
    dc_dv[0] -= - fxqs[0] * dS_dP;
#endif

    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[2];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[0] = - fxqs[2] * dS_dP;
#else
    dc_dv[0] -= - fxqs[2] * dS_dP;
#endif

    mvf = (mkqs[1] * moving->dim[1] + mjqs[0]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[1] = - fyqs[0] * dS_dP;
#else
    dc_dv[1] -= - fyqs[0] * dS_dP;
#endif

    mvf = (mkqs[1] * moving->dim[1] + mjqs[2]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[1] = - fyqs[2] * dS_dP;
#else
    dc_dv[1] -= - fyqs[2] * dS_dP;
#endif

    mvf = (mkqs[0] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[2] = - fzqs[0] * dS_dP;
#else
    dc_dv[2] -= - fzqs[0] * dS_dP;
#endif

    mvf = (mkqs[2] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
#if PLM_DONT_INVERT_GRADIENT
    dc_dv[2] = - fzqs[2] * dS_dP;
#else
    dc_dv[2] -= - fzqs[2] * dS_dP;
#endif

    dc_dv[0] = dc_dv[0] / moving->pix_spacing[0] / num_vox_f;
    dc_dv[1] = dc_dv[1] / moving->pix_spacing[1] / num_vox_f;
    dc_dv[2] = dc_dv[2] / moving->pix_spacing[2] / num_vox_f;
}
    

/* -----------------------------------------------------------------------
   Scoring functions
   ----------------------------------------------------------------------- */
/* NSh Mutual information version of implementation "C" with landmarks
See bspline_score_k_mse for details
Input files: 
-fixed.fcsv, moving.fcsv: fiducials files from Slicer3
-stiffness.txt: one line, stiffness of landmark-to-landmark spring
*/
static void
bspline_score_l_mi (BSPLINE_Parms *parms, 
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

    //begin NSh - new variables
    int i, num_landmarks;
    float landmark_fix[100][3], landmark_mov[100][3]; // in mm from Slicer fiducials
    int landvox_fix[100][3]; //fixed landmarks in voxels on subsampled fixed image
    float lmx, lmy, lmz;
    int lidx;
    FILE *fp1, *fp2;
    float land_score, land_coeff, land_grad_coeff, land_rawdist, l_dist;
    char s[1024], *s2;
    int land_sel, land_vis; //unused variables for reading in landmark files
    float voxel_score;
    // end NSh
	
	if (parms->debug) {
	sprintf (debug_fn, "dump_mi_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    //begin NSh
    fp1 = fopen("stiffness.txt","r");
    if (!fp1) { printf("cannot read stiffness.txt\n"); exit(1); }
    fscanf(fp1, "%f", &land_coeff);
    printf("landmark stiffness is %.2f\n", land_coeff);
    //land_coeff = 50000.;
    //land_coeff = 0.;
    fclose(fp1);

    fp1 = fopen("fixed.fcsv","r");
    if (!fp1) { printf("cannot read landmarks from fixed.fcsv\n"); exit(1); }
    i=0;
    while(!feof(fp1)) {
        fgets(s, 1024, fp1); if (feof(fp1)) break;
        if (s[0]=='#') continue; 
        s2=strchr(s,','); //skip the label field assuming it does not contain commas
        sscanf(s2, ",%f,%f,%f,%d,%d\n", &landmark_fix[i][0], &landmark_fix[i][1], &landmark_fix[i][2], &land_sel, &land_vis);
        i++;
    }
    fclose(fp1);
    printf("found %d landmarks on fixed image\n", i);
    num_landmarks = i;

    fp1 = fopen("moving.fcsv","r");
    if (!fp1) { printf("cannot read landmarks from moving.fcsv\n"); exit(1); }
    i=0;
    while(!feof(fp1)) {
        fgets(s, 1024, fp1); if (feof(fp1)) break;
        if (s[0]=='#') continue; 
        s2=strchr(s,','); //skip the label field assuming it does not contain commas
        sscanf(s2, ",%f,%f,%f,%d,%d\n", &landmark_mov[i][0], &landmark_mov[i][1], &landmark_mov[i][2], &land_sel, &land_vis);
        i++;
    }
    fclose(fp1);
    if (i!=num_landmarks) { printf("Error: different number of landmarks on fixed,moving images\n"); exit(1);}

    //position of fixed landmarks in voxels for vector field calculation
    for(i=0;i<num_landmarks;i++)
    { 
        landvox_fix[i][0] = floor( (-fixed->offset[0]-landmark_fix[i][0])/fixed->pix_spacing[0] );
        if (landvox_fix[i][0] < 0 || landvox_fix[i][0] >= fixed->dim[0]) 
            { printf("Error: landmark %d out of image!\n",i); exit(1);}

        landvox_fix[i][1] = floor( (-fixed->offset[1]-landmark_fix[i][1])/fixed->pix_spacing[1] );
        if (landvox_fix[i][1] < 0 || landvox_fix[i][1] >= fixed->dim[1]) 
            { printf("Error: landmark %d out of image!\n",i); exit(1);}

        // note the + sign in mm to voxel conversion for z-coordinate
        landvox_fix[i][2] = floor( (-fixed->offset[2]+landmark_fix[i][2])/fixed->pix_spacing[2] );
        if (landvox_fix[i][2] < 0 || landvox_fix[i][2] >= fixed->dim[2]) 
            { printf("Error: landmark %d out of image!\n",i); exit(1);}

        landmark_fix[i][0] = -fixed->offset[0]- landmark_fix[i][0];
        landmark_fix[i][1] = -fixed->offset[1]- landmark_fix[i][1];
        landmark_fix[i][2] = -fixed->offset[2]+ landmark_fix[i][2];

        landmark_mov[i][0] = -fixed->offset[0]- landmark_mov[i][0];
        landmark_mov[i][1] = -fixed->offset[1]- landmark_mov[i][1];
        landmark_mov[i][2] = -fixed->offset[2]+ landmark_mov[i][2];
    }

    land_score = 0;

    //end NSh


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
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
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
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
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


    //begin NSh - cost of landmark mismatch and its derivative
    voxel_score = ssd->score/num_vox;
    land_score = 0;
    land_rawdist = 0;
    land_grad_coeff = land_coeff * num_vox / num_landmarks;

    fp1  = fopen("warplist.fcsv","w");
    fp2 = fopen("distlist.dat","w");
    fprintf(fp1,"# name = warped\n");

    for(lidx=0;lidx<num_landmarks;lidx++)
    {

        p[0] = landvox_fix[lidx][0] / bxf->vox_per_rgn[0];
        q[0] = landvox_fix[lidx][0] % bxf->vox_per_rgn[0];
        p[1] = landvox_fix[lidx][1] / bxf->vox_per_rgn[1];
        q[1] = landvox_fix[lidx][1] % bxf->vox_per_rgn[1];
        p[2] = landvox_fix[lidx][2] / bxf->vox_per_rgn[2];
        q[2] = landvox_fix[lidx][2] % bxf->vox_per_rgn[2];

        qidx = INDEX_OF( q, bxf->vox_per_rgn);
        bspline_interp_pix (dxyz, bxf, p, qidx);

        lmx = landmark_mov[lidx][0] - dxyz[0];
        lmy = landmark_mov[lidx][1] - dxyz[1];
        lmz = landmark_mov[lidx][2] - dxyz[2];

        l_dist = (
            (landmark_fix[lidx][0]-lmx)*(landmark_fix[lidx][0]-lmx)+
            (landmark_fix[lidx][1]-lmy)*(landmark_fix[lidx][1]-lmy)+
            (landmark_fix[lidx][2]-lmz)*(landmark_fix[lidx][2]-lmz)
            );

        land_score += l_dist;
        land_rawdist += sqrt(l_dist);
        // calculating gradients
        dc_dv[0] =  +land_grad_coeff * (lmx-landmark_fix[lidx][0]); 
        dc_dv[1] =  +land_grad_coeff * (lmy-landmark_fix[lidx][1]); 
        dc_dv[2] =  +land_grad_coeff * (lmz-landmark_fix[lidx][2]); 
        bspline_update_grad (bst, bxf, p, qidx, dc_dv);

        fprintf(fp1, "W%d,%f,%f,%f,1,1\n", lidx,
		-fixed->offset[0]-lmx, 
		-fixed->offset[1]-lmy,  
		fixed->offset[2]+lmz );
        fprintf(fp2,"W%d %.3f\n", lidx, sqrt(l_dist));

    } //end for lidx
    fclose(fp1);
    fclose(fp2);

    printf("RAWDIST %.4f     VOXSCORE %.4f\n", land_rawdist/num_landmarks, voxel_score); 
    land_score = land_score * land_coeff *num_vox / num_landmarks;
    ssd->score += land_score;
    // end NSh

    mse_score = mse_score / num_vox;

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
    int rijk[3];
    int fijk[3], fv;
    float mijk[3];
    float fxyz[3];
    float mxyz[3];
    int mijk_f[3], mvf;      /* Floor */
    int mijk_r[3];           /* Round */
    int p[3];
    int q[3];
    float diff;
    float dc_dv[3];
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    float dxyz[3];
    int num_vox;
    float num_vox_f;
    int pidx, qidx;
    Timer timer;
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */
    float m_val;

    float mse_score = 0.0f;
    float* f_hist = mi_hist->f_hist;
    float* m_hist = mi_hist->m_hist;
    float* j_hist = mi_hist->j_hist;
#ifdef DOUBLE_HISTS
    double* f_hist_d = mi_hist->f_hist_d;
    double* m_hist_d = mi_hist->m_hist_d;
    double* j_hist_d = mi_hist->j_hist_d;
#endif
    static int it = 0;
    char debug_fn[1024];
    FILE* fp;
    int zz;

    if (parms->debug) {
	sprintf (debug_fn, "dump_mi_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(float));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(float));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins 
	* sizeof(float));
#ifdef DOUBLE_HISTS
    memset (f_hist_d, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist_d, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist_d, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
#endif
    num_vox = 0;

    /* PASS 1 - Accumulate histogram */
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

		/* Find correspondence in moving image */
		rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
		    dxyz, moving);

		/* If voxel is not inside moving image */
		if (!rc) continue;

		CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
		    li_1, li_2, moving);

		/* Find linear index of fixed image voxel */
		fv = INDEX_OF (fijk, fixed->dim);

		/* Find linear index of "corner voxel" in moving image */
		mvf = INDEX_OF (mijk_f, moving->dim);

		/* Compute moving image intensity using linear interpolation */
		/* Macro is slightly faster than function */
		BSPLINE_LI_VALUE (m_val, 
		    li_1[0], li_2[0],
		    li_1[1], li_2[1],
		    li_1[2], li_2[2],
		    mvf, m_img, moving);

#if defined (commentout)
		/* LINEAR INTERPOLATION */
		bspline_mi_hist_add (mi_hist, f_img[fv], m_val, 1.0);
#endif

#if defined (commentout)
		/* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
		bspline_mi_hist_add_pvi_8 (mi_hist, fixed, moving, 
		    fv, mvf, li_1, li_2);
#endif

		/* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
		bspline_mi_hist_add_pvi_6 (mi_hist, fixed, moving, 
		    fv, mvf, mijk);

		/* Compute intensity difference */
		diff = m_val - f_img[fv];
		mse_score += diff * diff;
		num_vox ++;
	    }
	}
    }

#ifdef DOUBLE_HISTS
    for (zz=0; zz < mi_hist->fixed.bins; zz++)
    	f_hist[zz] = (float)f_hist_d[zz];
    for (zz=0; zz < mi_hist->moving.bins; zz++)
    	m_hist[zz] = (float)m_hist_d[zz];
    for (zz=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++)
    	j_hist[zz] = (float)j_hist_d[zz];
#endif

    // Dump histogram images ??
    if (parms->xpm_hist_dump) {
	dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    if (parms->debug) {
	double tmp;
	tmp = 0;
	for (zz=0; zz < mi_hist->fixed.bins; zz++) { tmp += f_hist[zz]; }
	printf ("f_hist total: %f\n", tmp);
	tmp = 0;
	for (zz=0; zz < mi_hist->moving.bins; zz++) { tmp += m_hist[zz]; }
	printf ("m_hist total: %f\n", tmp);
	tmp = 0;
	for (zz=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
	    tmp += j_hist[zz];
	}
	printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->score = mi_hist_score (mi_hist, num_vox);
    num_vox_f = (float) num_vox;

    /* PASS 2 - Compute gradient */
    for (rijk[2] = 0, fijk[2] = bxf->roi_offset[2]; rijk[2] < bxf->roi_dim[2]; rijk[2]++, fijk[2]++) {
	p[2] = rijk[2] / bxf->vox_per_rgn[2];
	q[2] = rijk[2] % bxf->vox_per_rgn[2];
	fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];
	for (rijk[1] = 0, fijk[1] = bxf->roi_offset[1]; rijk[1] < bxf->roi_dim[1]; rijk[1]++, fijk[1]++) {
	    p[1] = rijk[1] / bxf->vox_per_rgn[1];
	    q[1] = rijk[1] % bxf->vox_per_rgn[1];
	    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
	    for (rijk[0] = 0, fijk[0] = bxf->roi_offset[0]; rijk[0] < bxf->roi_dim[0]; rijk[0]++, fijk[0]++) {
		int debug;
		int rc;

		debug = 0;

		p[0] = rijk[0] / bxf->vox_per_rgn[0];
		q[0] = rijk[0] % bxf->vox_per_rgn[0];
		fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];

		/* Get B-spline deformation vector */
		pidx = INDEX_OF (p, bxf->rdims);
		qidx = INDEX_OF (q, bxf->vox_per_rgn);
		bspline_interp_pix_b_inline (dxyz, bxf, pidx, qidx);

		/* Find linear index of fixed image voxel */
		fv = INDEX_OF (fijk, fixed->dim);

		/* Find correspondence in moving image */
		rc = bspline_find_correspondence (mxyz, mijk, fxyz, 
		    dxyz, moving);

		/* If voxel is not inside moving image */
		if (!rc) continue;

		/* LINEAR INTERPOLATION - (not implemented) */

#if defined (commentout)
		/* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
		bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
		    fv, mvf, mijk, num_vox_f);
#endif

		/* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
		bspline_mi_pvi_6_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
		    fv, mvf, mijk, num_vox_f);

		bspline_update_grad_b_inline (bst, bxf, pidx, qidx, dc_dv);
	    }
	}
    }

    if (parms->debug) {
	fclose (fp);
    }

    mse_score = mse_score / num_vox;

    report_score ("MI", bxf, bst, num_vox, plm_timer_report (&timer));
}

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

/* 
NSh Mean-square error registration with landmarks.
Based on bspline_score_a_mse

Input files: 
-fixed.fcsv, moving.fcsv: fiducials files from Slicer3
-stiffness.txt: one line, stiffness of landmark-to-landmark spring

Output files: warplist.fcsv is a fiducials file with landmarks 
on the warped image; distlist.dat are the distances between
corresponding landmarks in fixed and warped images.

Parameter: land_coeff = spring constant for attraction
between landmarks. land_coeff = 0 corresponds to no constraints
on landmarks, exactly as in bspline_score_a_mse.

Mar 15 2010 - NSh 
*/
void
bspline_score_k_mse 
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

    //begin NSh - new variables
    int num_landmarks;
    float landmark_fix[100][3], landmark_mov[100][3]; // in mm from Slicer fiducials
    int landvox_fix[100][3]; //fixed landmarks in voxels on subsampled fixed image
    float lmx, lmy, lmz;
    int lidx;
    FILE *fp, *fp2;
    float land_score, land_coeff, land_grad_coeff, land_rawdist, l_dist;
    char s[1024], *s2;
    int land_sel, land_vis; //unused variables for reading in landmark files
    float voxel_score;
    // end NSh

    plm_timer_start (&timer);

    //begin NSh
    fp = fopen("stiffness.txt","r");
    if (!fp) { printf("cannot read stiffness.txt\n"); exit(1); }
    fscanf(fp, "%f", &land_coeff);
    printf("landmark stiffness is %.2f\n", land_coeff);
    //land_coeff = 50000.;
    //land_coeff = 0.;
    fclose(fp);

    fp = fopen("fixed.fcsv","r");
    if (!fp) { printf("cannot read landmarks from fixed.fcsv\n"); exit(1); }
    i=0;
    while(!feof(fp)) {
        fgets(s, 1024, fp); if (feof(fp)) break;
        if (s[0]=='#') continue; 
        s2=strchr(s,','); //skip the label field assuming it does not contain commas
        sscanf(s2, ",%f,%f,%f,%d,%d\n", &landmark_fix[i][0], &landmark_fix[i][1], &landmark_fix[i][2], &land_sel, &land_vis);
        i++;
    }
    fclose(fp);
    printf("found %d landmarks on fixed image\n", i);
    num_landmarks = i;

    fp = fopen("moving.fcsv","r");
    if (!fp) { printf("cannot read landmarks from moving.fcsv\n"); exit(1); }
    i=0;
    while(!feof(fp)) {
        fgets(s, 1024, fp); if (feof(fp)) break;
        if (s[0]=='#') continue; 
        s2=strchr(s,','); //skip the label field assuming it does not contain commas
        sscanf(s2, ",%f,%f,%f,%d,%d\n", &landmark_mov[i][0], &landmark_mov[i][1], &landmark_mov[i][2], &land_sel, &land_vis);
        i++;
    }
    fclose(fp);
    if (i!=num_landmarks) { printf("Error: different number of landmarks on fixed,moving images\n"); exit(1);}

    //position of fixed landmarks in voxels for vector field calculation
    for(i=0;i<num_landmarks;i++)
    { 
        landvox_fix[i][0] = floor( (-fixed->offset[0]-landmark_fix[i][0])/fixed->pix_spacing[0] );
        if (landvox_fix[i][0] < 0 || landvox_fix[i][0] >= fixed->dim[0]) 
            { printf("Error: landmark %d out of image!\n",i); exit(1);}

        landvox_fix[i][1] = floor( (-fixed->offset[1]-landmark_fix[i][1])/fixed->pix_spacing[1] );
        if (landvox_fix[i][1] < 0 || landvox_fix[i][1] >= fixed->dim[1]) 
            { printf("Error: landmark %d out of image!\n",i); exit(1);}

        // note the + sign in mm to voxel conversion for z-coordinate
        landvox_fix[i][2] = floor( (-fixed->offset[2]+landmark_fix[i][2])/fixed->pix_spacing[2] );
        if (landvox_fix[i][2] < 0 || landvox_fix[i][2] >= fixed->dim[2]) 
				{ printf("Error: landmark %d out of image!\n",i); exit(1);}

        landmark_fix[i][0] = -fixed->offset[0]- landmark_fix[i][0];
        landmark_fix[i][1] = -fixed->offset[1]- landmark_fix[i][1];
        landmark_fix[i][2] = -fixed->offset[2]+ landmark_fix[i][2];

        landmark_mov[i][0] = -fixed->offset[0]- landmark_mov[i][0];
        landmark_mov[i][1] = -fixed->offset[1]- landmark_mov[i][1];
        landmark_mov[i][2] = -fixed->offset[2]+ landmark_mov[i][2];
    }

    land_score = 0;

    //end NSh

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

    //begin NSh - cost of landmark mismatch and its derivative
    voxel_score = ssd->score/num_vox;
    land_score = 0;
    land_rawdist = 0;
    land_grad_coeff = land_coeff * num_vox / num_landmarks;

    fp  = fopen("warplist.fcsv","w");
    fp2 = fopen("distlist.dat","w");
    fprintf(fp,"# name = warped\n");

    for(lidx=0;lidx<num_landmarks;lidx++)
    {

        p[0] = landvox_fix[lidx][0] / bxf->vox_per_rgn[0];
        q[0] = landvox_fix[lidx][0] % bxf->vox_per_rgn[0];
        p[1] = landvox_fix[lidx][1] / bxf->vox_per_rgn[1];
        q[1] = landvox_fix[lidx][1] % bxf->vox_per_rgn[1];
        p[2] = landvox_fix[lidx][2] / bxf->vox_per_rgn[2];
        q[2] = landvox_fix[lidx][2] % bxf->vox_per_rgn[2];

        qidx = INDEX_OF( q, bxf->vox_per_rgn);
        bspline_interp_pix (dxyz, bxf, p, qidx);

        lmx = landmark_mov[lidx][0] - dxyz[0];
        lmy = landmark_mov[lidx][1] - dxyz[1];
        lmz = landmark_mov[lidx][2] - dxyz[2];

        l_dist = (
        (landmark_fix[lidx][0]-lmx)*(landmark_fix[lidx][0]-lmx)+
        (landmark_fix[lidx][1]-lmy)*(landmark_fix[lidx][1]-lmy)+
        (landmark_fix[lidx][2]-lmz)*(landmark_fix[lidx][2]-lmz)
        );

        land_score += l_dist;
        land_rawdist += sqrt(l_dist);
        // calculating gradients
        dc_dv[0] =  +land_grad_coeff * (lmx-landmark_fix[lidx][0]); 
        dc_dv[1] =  +land_grad_coeff * (lmy-landmark_fix[lidx][1]); 
        dc_dv[2] =  +land_grad_coeff * (lmz-landmark_fix[lidx][2]); 
        bspline_update_grad (bst, bxf, p, qidx, dc_dv);

        fprintf(fp, "W%d,%f,%f,%f,1,1\n", lidx,
		-fixed->offset[0]-lmx, 
		-fixed->offset[1]-lmy,  
		fixed->offset[2]+lmz );
        fprintf(fp2,"W%d %.3f\n", lidx, sqrt(l_dist));

    } //end for lidx
    fclose(fp);
    fclose(fp2);

    printf("RAWDIST %.4f     VOXSCORE %.4f\n", land_rawdist/num_landmarks, voxel_score); 
    land_score = land_score * land_coeff *num_vox / num_landmarks;
    ssd->score += land_score;
    // end NSh

    /* Normalize score for MSE */
    ssd->score /= num_vox;
    for (i = 0; i < bxf->num_coeff; i++) {
	ssd->grad[i] /= num_vox;
	}

    interval = plm_timer_report (&timer);
    report_score ("MSE", bxf, bst, num_vox, interval);
}

////////////////////////////////////////////////////////////////////////////////
// FUNCTION: bspline_score_h_mse()
//
// This is a single core CPU implementation of CUDA implementation J.
// The tile "condense" method is demonstrated.
//
// ** This is the fastest know CPU implmentation for single core **
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
#if PLM_DONT_INVERT_GRADIENT
		    diff = m_val - f_img[idx_fixed];
#else
		    diff = f_img[idx_fixed] - m_val;
#endif

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
// ** This is the fastest know CPU implmentation for multi core **
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
#if PLM_DONT_INVERT_GRADIENT
		    diff = m_val - f_img[idx_fixed];
#else
		    diff = f_img[idx_fixed] - m_val;
#endif

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
//	    bspline_cuda_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'd':
//	    bspline_cuda_score_d_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'e':
//	    bspline_cuda_score_e_mse_v2 (parms, bst, bxf, fixed, moving, moving_grad);
	    //bspline_cuda_score_e_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'f':
//	    bspline_cuda_score_f_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'g':
//	    bspline_cuda_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'h':
//	    bspline_cuda_score_h_mse (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
//	    break;
	case 'i':
	    bspline_cuda_score_i_mse (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	case 'j':
	    bspline_cuda_score_j_mse (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
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
//	    bspline_score_a_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'b':
//	    bspline_score_b_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'c':
//	    bspline_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'd':
//	    bspline_score_d_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'e':
//	    bspline_score_e_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'f':
//	    bspline_score_f_mse (parms, bst, bxf, fixed, moving, moving_grad);
//	    break;
	case 'g':
	    bspline_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'h':
	    bspline_score_h_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'k':
	    bspline_score_k_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	}
    }

    if ((parms->threading == BTHR_CPU) && (parms->metric == BMET_MI)) {
	switch (parms->implementation) {
	case 'c':
	case 'd':
	    bspline_score_c_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'l':
	    bspline_score_l_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_score_c_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	}
    }


    /* Add vector field score/gradient to image score/gradient */
    if (parms->young_modulus) {
	printf ("comuting regularization\n");
	bspline_gradient_score (parms, bst, bxf, fixed, moving);
    }

    /* Add landmark score/gradient to image score/gradient */
    if (parms->landmarks) {
	printf ("comuting landmarks\n");
	bspline_landmarks_score (parms, bst, bxf, fixed, moving);
    }
    
}

void
bspline_run_optimization (
    BSPLINE_Xform* bxf, 
    Bspline_state **bst_in, 
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_state *bst;

    bst = bspline_state_create (bxf, parms, fixed, moving, moving_grad);
    log_parms (parms);
    log_bxf_header (bxf);

    if (parms->metric == BMET_MI) {
	bspline_initialize_mi (parms, fixed, moving);
    }

    /* Do the optimization */
    bspline_optimize (bxf, bst, parms, fixed, moving, moving_grad);

    if (bst_in) {
	*bst_in = bst;
    } else {
	bspline_state_destroy (bst);
    }
}
