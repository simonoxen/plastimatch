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
#include "bspline_mse_cpu_c.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif
#include "bspline_regularize.h"
#include "bspline_landmarks.h"
#include "bspline_macros.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "logfile.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_path.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xpm.h"
#include "delayload.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif


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
bspline_parms_set_default (Bspline_parms* parms)
{
    memset (parms, 0, sizeof(Bspline_parms));
    parms->threading = BTHR_CPU;
    parms->optimization = BOPT_LBFGSB;
    parms->metric = BMET_MSE;
    parms->implementation = '\0';
    parms->max_its = 10;
    parms->max_feval = 10;
    parms->convergence_tol = 0.1;
    parms->convergence_tol_its = 4;
    parms->debug = 0;
    parms->lbfgsb_factr = 1.0e+7;
    parms->lbfgsb_pgtol = 1.0e-5;

#if defined (commentout)
    parms->landmarks = 0;
    parms->landmark_stiffness = 1.0;
    parms->landmark_implementation = 'a';
    parms->young_modulus = 0.0;
    parms->rbf_radius = 0.0;
    parms->rbf_young_modulus = 0.0;
#endif

    parms->mi_hist.f_hist = 0;
    parms->mi_hist.m_hist = 0;
    parms->mi_hist.j_hist = 0;

    parms->mi_hist.fixed.bins = 20;
    parms->mi_hist.moving.bins = 20;
    parms->mi_hist.joint.bins = parms->mi_hist.fixed.bins
	* parms->mi_hist.moving.bins;

    parms->mi_hist.fixed.big_bin = 0;
    parms->mi_hist.moving.big_bin = 0;
    parms->mi_hist.joint.big_bin = 0;

    parms->gpuid = 0;
    parms->gpu_zcpy = 0;
}

void
bspline_xform_set_default (Bspline_xform* bxf)
{
    int d;

    memset (bxf, 0, sizeof (Bspline_xform));

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
    Bspline_xform* bxf,
    Bspline_parms *parms,
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
#if (CUDA_FOUND)
    Dev_Pointers_Bspline* dev_ptrs 
        = (Dev_Pointers_Bspline*) malloc (sizeof (Dev_Pointers_Bspline));

    bst->dev_ptrs = dev_ptrs;
    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {
        /* Be sure we loaded the CUDA plugin */
        if (!delayload_cuda ()) { exit (0); }
        LOAD_LIBRARY (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mse_init_j, libplmcuda);

        switch (parms->implementation) {
        case 'j':
        case '\0':   /* Default */
            CUDA_bspline_mse_init_j (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        default:
            printf ("Warning: option -f %c unavailble.  Switching to -f j\n",
                    parms->implementation);
            CUDA_bspline_mse_init_j (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        }

        UNLOAD_LIBRARY (libplmcuda);
    } 
    else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {

        /* Be sure we loaded the CUDA plugin */
        if (!delayload_cuda ()) { exit (0); }
        LOAD_LIBRARY (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mi_init_a, libplmcuda);

        switch (parms->implementation) {
        case 'a':
            CUDA_bspline_mi_init_a (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        default:
            printf ("Warning: option -f %c unavailble.  Defaulting to -f a\n",
                parms->implementation);
            CUDA_bspline_mi_init_a (dev_ptrs, fixed, moving, moving_grad, bxf, parms);
            break;
        }

        UNLOAD_LIBRARY (libplmcuda);
    }
    else {
        printf ("No cuda initialization performed.\n");
    }
#endif
}

Bspline_state *
bspline_state_create (
    Bspline_xform *bxf, 
    Bspline_parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    Bspline_state *bst = (Bspline_state*) malloc (sizeof (Bspline_state));
    memset (bst, 0, sizeof (Bspline_state));
    bst->ssd.grad = (float*) malloc (bxf->num_coeff * sizeof(float));
    memset (bst->ssd.grad, 0, bxf->num_coeff * sizeof(float));

    bspline_cuda_state_create (bst, bxf, parms, fixed, moving, moving_grad);

    if (parms->metric == BMET_MI) {
        int i;
        for (i = 0; i < bxf->num_coeff; i++) {
            bxf->coeff[i] = 0.5f;
        }
    }

    return bst;
}

void
write_bxf (char* filename, Bspline_xform* bxf)
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

Bspline_xform* 
read_bxf (char* filename)
{
    Bspline_xform* bxf;
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
    bxf = (Bspline_xform*) malloc (sizeof(Bspline_xform));
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
log_parms (Bspline_parms* parms)
{
    logfile_printf ("BSPLINE PARMS\n");
    logfile_printf ("max_its = %d\n", parms->max_its);
    logfile_printf ("max_feval = %d\n", parms->max_feval);
}

static void
log_bxf_header (Bspline_xform* bxf)
{
    logfile_printf ("BSPLINE XFORM HEADER\n");
    logfile_printf ("vox_per_rgn = %d %d %d\n", bxf->vox_per_rgn[0], bxf->vox_per_rgn[1], bxf->vox_per_rgn[2]);
    logfile_printf ("roi_offset = %d %d %d\n", bxf->roi_offset[0], bxf->roi_offset[1], bxf->roi_offset[2]);
    logfile_printf ("roi_dim = %d %d %d\n", bxf->roi_dim[0], bxf->roi_dim[1], bxf->roi_dim[2]);
}

void
dump_gradient (Bspline_xform* bxf, BSPLINE_Score* ssd, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bxf->num_coeff; i++) {
	fprintf (fp, "%f\n", ssd->grad[i]);
    }
    fclose (fp);
}

void
dump_coeff (Bspline_xform* bxf, char* fn)
{
    int i;
    FILE* fp = fopen (fn,"wb");
    for (i = 0; i < bxf->num_coeff; i++) {
	fprintf (fp, "%f\n", bxf->coeff[i]);
    }
    fclose (fp);
}

void
dump_luts (Bspline_xform* bxf)
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
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
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
bspline_display_coeff_stats (Bspline_xform* bxf)
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
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform* bxf
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
bspline_set_coefficients (Bspline_xform* bxf, float val)
{
    int i;

    for (i = 0; i < bxf->num_coeff; i++) {
	bxf->coeff[i] = val;
    }
}

void
bspline_xform_initialize 
(
 Bspline_xform* bxf,         /* Output: bxf is initialized */
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

/* -----------------------------------------------------------------------
   This extends the bspline grid.  Note, that the new roi_offset 
    in the bxf will not be the same as the one requested, because 
    bxf routines implicitly require that the first voxel of the 
    ROI matches the position of the control point. 
   ----------------------------------------------------------------------- */
/* GCS -- Is there an implicit assumption that the roi_origin > 0? */
void
bspline_xform_extend (
    Bspline_xform* bxf,	     /* Output: bxf is initialized */
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
bspline_initialize_mi_bigbin (double* hist, BSPLINE_MI_Hist_Parms* hparms, Volume* vol)
{
    int i, idx_bin;
    float* img = (float*) vol->img;

    if (!img) {
        logfile_printf ("ERROR: trying to pre-scan empty image!\n");
        exit (-1);
    }

    /* build a quick histogram */
    for (i=0; i<vol->npix; i++) {
        idx_bin = floor ((img[i] - hparms->offset) / hparms->delta);
        hist[idx_bin]++;
    }

    /* look for biggest bin */
    for(i=0; i<hparms->bins; i++) {
        if (hist[i] > hist[hparms->big_bin]) {
            hparms->big_bin = i;
        }
    }
//    printf ("big_bin: %i\n", hparms->big_bin);
    
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
bspline_initialize_mi (Bspline_parms* parms, Volume* fixed, Volume* moving)
{
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    mi_hist->m_hist = (double*) malloc (sizeof (double) * mi_hist->moving.bins);
    mi_hist->f_hist = (double*) malloc (sizeof (double) * mi_hist->fixed.bins);
    mi_hist->j_hist = (double*) malloc (sizeof (double) * mi_hist->fixed.bins * mi_hist->moving.bins);
    bspline_initialize_mi_vol (&mi_hist->moving, moving);
    bspline_initialize_mi_vol (&mi_hist->fixed, fixed);

    /* Initialize biggest bin trackers for OpenMP MI */
    bspline_initialize_mi_bigbin (mi_hist->f_hist, &mi_hist->fixed, fixed);
    bspline_initialize_mi_bigbin (mi_hist->m_hist, &mi_hist->moving, moving);

    /* This estimate /could/ be wrong for certain image sets */
    /* Will be auto corrected after first evaluation if incorrect */
    mi_hist->joint.big_bin = mi_hist->fixed.big_bin
                           * mi_hist->moving.bins
                           + mi_hist->moving.big_bin;
}

void
bspline_xform_free (Bspline_xform* bxf)
{
    free (bxf->coeff);
    free (bxf->q_lut);
    free (bxf->c_lut);
}

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

void
bspline_parms_free (Bspline_parms* parms)
{
    if (parms->mi_hist.j_hist) {
	free (parms->mi_hist.f_hist);
	free (parms->mi_hist.m_hist);
	free (parms->mi_hist.j_hist);
    }
}

void
bspline_state_destroy (
    Bspline_state* bst,
    Bspline_parms *parms, 
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad
)
{
    if (bst->ssd.grad) {
        free (bst->ssd.grad);
    }

#if (CUDA_FOUND)
    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {
        /* Be sure we loaded the CUDA plugin! */
        if (!delayload_cuda ()) { exit (0); }

        // JAS 10.27.2010
        // CUDA zero-paging could have replaced the fixed, moving, or moving_grad
        // pointers with pointers to pinned CPU memory, which must be freed using
        // cudaFreeHost().  So, to prevent a segfault, we must free and NULL
        // these pointers before they are attempted to be free()ed in the standard
        // fashion.  Remember, free(NULL) is okay!
        LOAD_LIBRARY (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mse_cleanup_j, libplmcuda);
        CUDA_bspline_mse_cleanup_j (bst->dev_ptrs, fixed, moving, moving_grad);
        UNLOAD_LIBRARY (libplmcuda);
    }
    else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {
        /* Be sure we loaded the CUDA plugin! */
        if (!delayload_cuda ()) { exit (0); }

        LOAD_LIBRARY (libplmcuda);
        LOAD_SYMBOL (CUDA_bspline_mi_cleanup_a, libplmcuda);
        CUDA_bspline_mi_cleanup_a (bst->dev_ptrs, fixed, moving, moving_grad);
        UNLOAD_LIBRARY (libplmcuda);
    }
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
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
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
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;

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


/* This algorithm uses a un-normalized score. */
static float
mi_hist_score_omp (BSPLINE_MI_Hist* mi_hist, int num_vox)
{
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;

    int f_bin, m_bin, j_bin;
    double fnv = (double) num_vox;
    double score = 0;
    double hist_thresh = 0.001 / (mi_hist->moving.bins * mi_hist->fixed.bins);

    /* Compute cost */
#pragma omp parallel for reduction(-:score)
    for (j_bin=0; j_bin < (mi_hist->fixed.bins * mi_hist->moving.bins); j_bin++) {
        m_bin = j_bin % mi_hist->moving.bins;
        f_bin = j_bin / mi_hist->moving.bins;
        
        if (j_hist[j_bin] > hist_thresh) {
            score -= j_hist[j_bin] * logf(fnv * j_hist[j_bin] / (m_hist[m_bin] * f_hist[f_bin]));
        }
    }

    score = score / fnv;
    return (float) score;
}

void
bspline_interp_pix (float out[3], Bspline_xform* bxf, int p[3], int qidx)
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

void
bspline_interp_pix_b (
    float out[3], 
    Bspline_xform* bxf, 
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
			Bspline_xform* bxf)
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
    Bspline_xform* bxf, 
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
bspline_update_grad_b_inline (Bspline_state* bst, Bspline_xform* bxf, 
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
bspline_update_grad_b (Bspline_state* bst, Bspline_xform* bxf, 
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
bspline_update_sets (float* sets_x, float* sets_y, float* sets_z,
                          int qidx, float* dc_dv, Bspline_xform* bxf)
{
    int sidx;   // set index

    /* Initialize q_lut */
    float* q_lut = &bxf->q_lut[64*qidx];

    /* Condense dc_dv & commit to sets for tile */
    for (sidx=0; sidx<64; sidx++) {
        sets_x[sidx] += dc_dv[0] * q_lut[sidx];
        sets_y[sidx] += dc_dv[1] * q_lut[sidx];
        sets_z[sidx] += dc_dv[2] * q_lut[sidx];
    }
}


void
bspline_sort_sets (float* cond_x, float* cond_y, float* cond_z,
                        float* sets_x, float* sets_y, float* sets_z,
                        int pidx, Bspline_xform* bxf)
{
        int sidx, kidx;
        int* k_lut = (int*)malloc(64*sizeof(int));

        /* Generate the knot index lut */
        find_knots (k_lut, pidx, bxf->rdims, bxf->cdims);

        /* Rackem' Up */
        for (sidx=0; sidx<64; sidx++) {
            kidx = k_lut[sidx];

            cond_x[ (64*kidx) + sidx ] = sets_x[sidx];
            cond_y[ (64*kidx) + sidx ] = sets_y[sidx];
            cond_z[ (64*kidx) + sidx ] = sets_z[sidx];
        }

        free (k_lut);
}


void
bspline_make_grad (float* cond_x, float* cond_y, float* cond_z,
                   Bspline_xform* bxf, BSPLINE_Score* ssd)
{
    int kidx, sidx;

    for (kidx=0; kidx < (bxf->cdims[0] * bxf->cdims[1] * bxf->cdims[2]); kidx++) {
        for (sidx=0; sidx<64; sidx++) {
            ssd->grad[3*kidx + 0] += cond_x[64*kidx + sidx];
            ssd->grad[3*kidx + 1] += cond_y[64*kidx + sidx];
            ssd->grad[3*kidx + 2] += cond_z[64*kidx + sidx];
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
    float ma,           /*  Input: (Unrounded) pixel coordinate (in vox) */
    int dmax,		/*  Input: Maximum coordinate in this dimension */
    int* maf,		/* Output: x, y, or z coord of "floor" pixel in moving img */
    int* mar,		/* Output: x, y, or z coord of "round" pixel in moving img */
    float* fa1,		/* Output: Fraction of interpolant for lower index voxel */
    float* fa2		/* Output: Fraction of interpolant for upper index voxel */
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

    faqs[2] = t22;
    faqs[1] = - t2 + t + 0.5;
    faqs[0] = t22 - t + 0.5;

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
    double* j_hist, 
    double* f_hist, 
    double* m_hist, 
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
report_score (
    char *alg, 
    Bspline_xform *bxf, 
    Bspline_state *bst, 
    int num_vox, 
    double timing)
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
    // if the optimizer is performing adequately.
    if (!strcmp (alg, "MI")) {
	logfile_printf (
	    "%s[%2d,%3d] %1.8f NV %6d GM %9.3f GN %9.3f [%9.3f secs]\n", 
	    alg, bst->it, bst->feval, bst->ssd.score, num_vox, ssd_grad_mean, 
	    ssd_grad_norm, timing);
    } else {
	logfile_printf (
	    "%s[%2d,%3d] %9.3f NV %6d GM %9.3f GN %9.3f [%9.3f secs]\n", 
	    alg, bst->it, bst->feval, bst->ssd.score, num_vox, ssd_grad_mean, 
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
	
    double *m_hist = mi_hist->m_hist;
    double *f_hist = mi_hist->f_hist;
    double *j_hist = mi_hist->j_hist;
	
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
int
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

/* JAS 2010.11.30
 * This is an intentionally bad idea and will be removed as soon the paper I'm
 * writing sees some ink.
 *
 * Uses CRITICAL SECTIONS instead of locks to make histogram writes thread
 * safe when employing multi-core */
#if (OPENMP_FOUND)
static inline void
bspline_mi_hist_add_pvi_8_omp_crits (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float li_1[3],           /* Fraction of interpolant in lower index */
    float li_2[3])           /* Fraction of interpolant in upper index */
{
    float w[8];
    int n[8];
    int idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double *f_hist = mi_hist->f_hist;
    double *m_hist = mi_hist->m_hist;
    double *j_hist = mi_hist->j_hist;


    /* Compute partial volumes from trilinear interpolation weights */
    w[0] = li_1[0] * li_1[1] * li_1[2];	// Partial Volume w0
    w[1] = li_2[0] * li_1[1] * li_1[2];	// Partial Volume w1
    w[2] = li_1[0] * li_2[1] * li_1[2];	// Partial Volume w2
    w[3] = li_2[0] * li_2[1] * li_1[2];	// Partial Volume w3
    w[4] = li_1[0] * li_1[1] * li_2[2];	// Partial Volume w4
    w[5] = li_2[0] * li_1[1] * li_2[2];	// Partial Volume w5
    w[6] = li_1[0] * li_2[1] * li_2[2];	// Partial Volume w6
    w[7] = li_2[0] * li_2[1] * li_2[2];	// Partial Volume w7

    /* Note that Sum(wN) for N within [0,7] should = 1 */

    // Calculate Point Indices for 8 neighborhood
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    // Calculate fixed histogram bin and increment it
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);

    #pragma omp critical (fixed_histogram)
    {
        f_hist[idx_fbin]++;
    }

    offset_fbin = idx_fbin * mi_hist->moving.bins;

    // Add PV weights to moving & joint histograms   
    for (idx_pv=0; idx_pv<8; idx_pv++) {

        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        idx_jbin = offset_fbin + idx_mbin;

        if (idx_mbin != mi_hist->moving.big_bin) {
            #pragma omp critical (moving_histogram)
            {
                m_hist[idx_mbin] += w[idx_pv];
            }
        }

        if (idx_jbin != mi_hist->joint.big_bin) {
            #pragma omp critical (joint_histogram)
            {
                j_hist[idx_jbin] += w[idx_pv];
            }
        }
    }
}
#endif

/* Used locks to make histogram writes
 * thread safe when employing multi-core */
#if (OPENMP_FOUND)
static inline void
bspline_mi_hist_add_pvi_8_omp (
    BSPLINE_MI_Hist* mi_hist, 
    Volume *fixed, 
    Volume *moving, 
    int fv, 
    int mvf, 
    float li_1[3],           /* Fraction of interpolant in lower index */
    float li_2[3],           /* Fraction of interpolant in upper index */
    omp_lock_t* f_locks,
    omp_lock_t* m_locks,
    omp_lock_t* j_locks)
{
    float w[8];
    int n[8];
    int idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double *f_hist = mi_hist->f_hist;
    double *m_hist = mi_hist->m_hist;
    double *j_hist = mi_hist->j_hist;


    /* Compute partial volumes from trilinear interpolation weights */
    w[0] = li_1[0] * li_1[1] * li_1[2];	// Partial Volume w0
    w[1] = li_2[0] * li_1[1] * li_1[2];	// Partial Volume w1
    w[2] = li_1[0] * li_2[1] * li_1[2];	// Partial Volume w2
    w[3] = li_2[0] * li_2[1] * li_1[2];	// Partial Volume w3
    w[4] = li_1[0] * li_1[1] * li_2[2];	// Partial Volume w4
    w[5] = li_2[0] * li_1[1] * li_2[2];	// Partial Volume w5
    w[6] = li_1[0] * li_2[1] * li_2[2];	// Partial Volume w6
    w[7] = li_2[0] * li_2[1] * li_2[2];	// Partial Volume w7

    /* Note that Sum(wN) for N within [0,7] should = 1 */

    // Calculate Point Indices for 8 neighborhood
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    // Calculate fixed histogram bin and increment it
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);

    omp_set_lock(&f_locks[idx_fbin]);
    f_hist[idx_fbin]++;
    omp_unset_lock(&f_locks[idx_fbin]);

    offset_fbin = idx_fbin * mi_hist->moving.bins;

    // Add PV weights to moving & joint histograms   
    for (idx_pv=0; idx_pv<8; idx_pv++) {

        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        idx_jbin = offset_fbin + idx_mbin;

        if (idx_mbin != mi_hist->moving.big_bin) {
            omp_set_lock(&m_locks[idx_mbin]);
            m_hist[idx_mbin] += w[idx_pv];
            omp_unset_lock(&m_locks[idx_mbin]);
        }

        if (idx_jbin != mi_hist->joint.big_bin) {
            omp_set_lock(&j_locks[idx_jbin]);
            j_hist[idx_jbin] += w[idx_pv];
            omp_unset_lock(&j_locks[idx_jbin]);
        }
    }
}
#endif


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
    float w[8];
    int n[8];
    int idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double *f_hist = mi_hist->f_hist;
    double *m_hist = mi_hist->m_hist;
    double *j_hist = mi_hist->j_hist;


    /* Compute partial volumes from trilinear interpolation weights */
    w[0] = li_1[0] * li_1[1] * li_1[2];	// Partial Volume w0
    w[1] = li_2[0] * li_1[1] * li_1[2];	// Partial Volume w1
    w[2] = li_1[0] * li_2[1] * li_1[2];	// Partial Volume w2
    w[3] = li_2[0] * li_2[1] * li_1[2];	// Partial Volume w3
    w[4] = li_1[0] * li_1[1] * li_2[2];	// Partial Volume w4
    w[5] = li_2[0] * li_1[1] * li_2[2];	// Partial Volume w5
    w[6] = li_1[0] * li_2[1] * li_2[2];	// Partial Volume w6
    w[7] = li_2[0] * li_2[1] * li_2[2];	// Partial Volume w7

    /* Note that Sum(wN) for N within [0,7] should = 1 */

    // Calculate Point Indices for 8 neighborhood
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    // Calculate fixed histogram bin and increment it
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    f_hist[idx_fbin]++;

    offset_fbin = idx_fbin * mi_hist->moving.bins;

    // Add PV weights to moving & joint histograms   
    for (idx_pv=0; idx_pv<8; idx_pv++) {
        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        idx_jbin = offset_fbin + idx_mbin;
        m_hist[idx_mbin] += w[idx_pv];
        j_hist[idx_jbin] += w[idx_pv];
    }

}

#if defined (commentout)
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
#endif

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
    float num_vox_f,               /* Input */
    float li_1[3],                 /* Input */
    float li_2[3]                  /* Input */
)
{
    float dS_dP;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
        
    BSPLINE_Score* ssd = &bst->ssd;
    int idx_fbin, idx_mbin, idx_jbin, idx_pv;
    int offset_fbin;
    int n[8];
    float dw[24];

    dc_dv[0] = dc_dv[1] = dc_dv[2] = 0.0f;

    /* Calculate Point Indices for 8 neighborhood */
    n[0] = mvf;
    n[1] = n[0] + 1;
    n[2] = n[0] + moving->dim[0];
    n[3] = n[2] + 1;
    n[4] = n[0] + moving->dim[0]*moving->dim[1];
    n[5] = n[4] + 1;
    n[6] = n[4] + moving->dim[0];
    n[7] = n[6] + 1;

    /* Pre-compute differential PV slices */
    dw[3*0+0] = (  -1 ) * li_1[1] * li_1[2];    // dw0
    dw[3*0+1] = li_1[0] * (  -1 ) * li_1[2];
    dw[3*0+2] = li_1[0] * li_1[1] * (  -1 );

    dw[3*1+0] = (  +1 ) * li_1[1] * li_1[2];    // dw1
    dw[3*1+1] = li_2[0] * (  -1 ) * li_1[2];
    dw[3*1+2] = li_2[0] * li_1[1] * (  -1 );

    dw[3*2+0] = (  -1 ) * li_2[1] * li_1[2];    // dw2
    dw[3*2+1] = li_1[0] * (  +1 ) * li_1[2];
    dw[3*2+2] = li_1[0] * li_2[1] * (  -1 );

    dw[3*3+0] = (  +1 ) * li_2[1] * li_1[2];    // dw3
    dw[3*3+1] = li_2[0] * (  +1 ) * li_1[2];
    dw[3*3+2] = li_2[0] * li_2[1] * (  -1 );

    dw[3*4+0] = (  -1 ) * li_1[1] * li_2[2];    // dw4
    dw[3*4+1] = li_1[0] * (  -1 ) * li_2[2];
    dw[3*4+2] = li_1[0] * li_1[1] * (  +1 );

    dw[3*5+0] = (  +1 ) * li_1[1] * li_2[2];    // dw5
    dw[3*5+1] = li_2[0] * (  -1 ) * li_2[2];
    dw[3*5+2] = li_2[0] * li_1[1] * (  +1 );

    dw[3*6+0] = (  -1 ) * li_2[1] * li_2[2];    // dw6
    dw[3*6+1] = li_1[0] * (  +1 ) * li_2[2];
    dw[3*6+2] = li_1[0] * li_2[1] * (  +1 );

    dw[3*7+0] = (  +1 ) * li_2[1] * li_2[2];    // dw7
    dw[3*7+1] = li_2[0] * (  +1 ) * li_2[2];
    dw[3*7+2] = li_2[0] * li_2[1] * (  +1 );


    /* Fixed image voxel's histogram index */
    idx_fbin = floor ((f_img[fv] - mi_hist->fixed.offset) / mi_hist->fixed.delta);
    offset_fbin = idx_fbin * mi_hist->moving.bins;

    /* Partial Volume Contributions */
    for (idx_pv=0; idx_pv<8; idx_pv++) {
        idx_mbin = floor ((m_img[n[idx_pv]] - mi_hist->moving.offset) / mi_hist->moving.delta);
        idx_jbin = offset_fbin + idx_mbin;
        if (j_hist[idx_jbin] > 0.0001) {
        	dS_dP = logf((num_vox_f * j_hist[idx_jbin]) / (f_hist[idx_fbin] * m_hist[idx_mbin])) - ssd->score;
        	dc_dv[0] -= dw[3*idx_pv+0] * dS_dP;
        	dc_dv[1] -= dw[3*idx_pv+1] * dS_dP;
        	dc_dv[2] -= dw[3*idx_pv+2] * dS_dP;
        }
    }

    dc_dv[0] = dc_dv[0] / num_vox_f / moving->pix_spacing[0];
    dc_dv[1] = dc_dv[1] / num_vox_f / moving->pix_spacing[1];
    dc_dv[2] = dc_dv[2] / num_vox_f / moving->pix_spacing[2];


#if defined (commentout)
    for (idx_pv=0; idx_pv<8; idx_pv++) {
        printf ("dw%i [ %2.5f %2.5f %2.5f ]\n", idx_pv, dw[3*idx_pv+0], dw[3*idx_pv+1], dw[3*idx_pv+2]);
    }

    printf ("S [ %2.5f %2.5f %2.5f ]\n\n\n", dc_dv[0], dc_dv[1], dc_dv[2]);
    exit(0);
#endif
}

static inline void
bspline_mi_pvi_6_dc_dv (
    float dc_dv[3],                /* Output */
    BSPLINE_MI_Hist* mi_hist,      /* Input */
    Bspline_state *bst,            /* Input */
    Volume *fixed,                 /* Input */
    Volume *moving,                /* Input */
    int fv,                        /* Input: Index into fixed  image */
    int mvf,                       /* Input: Index into moving image (unnecessary) */
    float mijk[3],                 /* Input: ijk indices in moving image (vox) */
    float num_vox_f                /* Input: Number of voxels falling into the moving image */
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
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
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
    dc_dv[0] += - fxqs[1] * dS_dP;
    dc_dv[1] += - fyqs[1] * dS_dP;
    dc_dv[2] += - fzqs[1] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[0];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
    dc_dv[0] += - fxqs[0] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[2];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
    dc_dv[0] += - fxqs[2] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[0]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
    dc_dv[1] += - fyqs[0] * dS_dP;

    mvf = (mkqs[1] * moving->dim[1] + mjqs[2]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
    dc_dv[1] += - fyqs[2] * dS_dP;

    mvf = (mkqs[0] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
    dc_dv[2] += - fzqs[0] * dS_dP;

    mvf = (mkqs[2] * moving->dim[1] + mjqs[1]) * moving->dim[0] + miqs[1];
    bspline_mi_hist_lookup (j_idxs, m_idxs, f_idxs, fxs, 
	mi_hist, f_img[fv], m_img[mvf]);
    dS_dP = compute_dS_dP (j_hist, f_hist, m_hist, j_idxs, f_idxs, m_idxs, 
	num_vox_f, fxs, ssd->score, debug);
    dc_dv[2] += - fzqs[2] * dS_dP;

    dc_dv[0] = dc_dv[0] / moving->pix_spacing[0] / num_vox_f;
    dc_dv[1] = dc_dv[1] / moving->pix_spacing[1] / num_vox_f;
    dc_dv[2] = dc_dv[2] / moving->pix_spacing[2] / num_vox_f;
}
    

/* -----------------------------------------------------------------------
   Scoring functions
   ----------------------------------------------------------------------- */

/* JAS 2010.11.30
 * This is an intentionally bad idea and will be removed as soon the paper I'm
 * writing sees some ink.
 * 
 * B-Spline Registration using Mutual Information
 * Implementation F (not good... only for comparison to E)
 *   -- Histograms are OpenMP accelerated
 *      (using CRITICAL SECTIONS... just to show better performance with locks)
 *   -- Uses OpenMP for Cost & dc_dv computation
 *   -- Uses methods introduced in bspline_score_g_mse
 *        to compute dc_dp more rapidly.
 */
#if (OPENMP_FOUND)
static void
bspline_score_f_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    int pidx;
    int num_vox;
    float num_vox_f;
    Timer timer;

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    static int it = 0;
    double mhis = 0.0f;      /* Moving histogram incomplete sum */
    double jhis = 0.0f;      /* Joint  histogram incomplete sum */
    char debug_fn[1024];
    FILE* fp;
    int i, j, zz;

    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);


    if (parms->debug) {
        sprintf (debug_fn, "dump_mi_%02d.txt", it++);
        fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);
    num_vox = 0;

    /* PASS 1 - Accumulate histogram */
#pragma omp parallel for
    for (pidx=0; pidx < num_tiles; pidx++) {
        int rc;
        int fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        int mijk_f[3], mvf;      /* Floor */
        int mijk_r[3];           /* Round */
        int p[3];
        int q[3];
        float dxyz[3];
        int qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        for (q[2]=0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
            for (q[1]=0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
                for (q[0]=0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
                    
                    /* Construct coordinates into fixed image volume */
                    fijk[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0]*p[0] + q[0];
                    fijk[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1]*p[1] + q[1];
                    fijk[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2]*p[2] + q[2];
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];
                    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
                    fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];

                    /* Construct the linear index within tile space */
                    qidx = INDEX_OF (q, bxf->vox_per_rgn);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
                                                 li_1, li_2,
                                                 moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = INDEX_OF (fijk, fixed->dim);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = INDEX_OF (mijk_f, moving->dim);

                    /* Add to histogram */

                    bspline_mi_hist_add_pvi_8_omp_crits (mi_hist, fixed, moving, 
                                                   fv, mvf, li_1, li_2);
                }
            }
        }   // tile
    }   // openmp

    /* Compute num_vox and find fullest fixed hist bin */
    for(i=0; i<mi_hist->fixed.bins; i++) {
        if (f_hist[i] > f_hist[mi_hist->fixed.big_bin]) {
            mi_hist->fixed.big_bin = i;
        }
        num_vox += f_hist[i];
    }

    /* Fill in the missing histogram bin */
    for(i=0; i<mi_hist->moving.bins; i++) {
        mhis += m_hist[i];
    }
    m_hist[mi_hist->moving.big_bin] = (double)num_vox - mhis;


    /* Look for the biggest moving histogram bin */
    for(i=0; i<mi_hist->moving.bins; i++) {
        if (m_hist[i] > m_hist[mi_hist->moving.big_bin]) {
            mi_hist->moving.big_bin = i;
        }
    }


    /* Fill in the missing jnt hist bin */
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            jhis += j_hist[j*mi_hist->moving.bins + i];
        }
    }
    j_hist[mi_hist->joint.big_bin] = (double)num_vox - jhis;

    
    /* Look for the biggest joint histogram bin */
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            if (j_hist[j*mi_hist->moving.bins + i] > j_hist[mi_hist->joint.big_bin]) {
                mi_hist->joint.big_bin = j*mi_hist->moving.bins + i;
            }
        }
    }

    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->score = mi_hist_score_omp (mi_hist, num_vox);
    num_vox_f = (float) num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    for (pidx=0; pidx < num_tiles; pidx++) {
        int rc;
        int fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        int mijk_f[3], mvf;      /* Floor */
        int mijk_r[3];           /* Round */
        int p[3];
        int q[3];
        float dxyz[3];
        int qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        for (q[2]=0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
            for (q[1]=0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
                for (q[0]=0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
                    
                    /* Construct coordinates into fixed image volume */
                    fijk[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0]*p[0] + q[0];
                    fijk[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1]*p[1] + q[1];
                    fijk[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2]*p[2] + q[2];
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];
                    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
                    fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];

                    /* Construct the linear index within tile space */
                    qidx = INDEX_OF (q, bxf->vox_per_rgn);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
                                                 li_1, li_2,
                                                 moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = INDEX_OF (fijk, fixed->dim);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = INDEX_OF (mijk_f, moving->dim);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                        fv, mvf, mijk, num_vox_f, li_1, li_2);

                    /* Update condensed tile sets */
                    bspline_update_sets (sets_x, sets_y, sets_z,
                                         qidx, dc_dv, bxf);
                }
            }
        }   // tile

        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
                           sets_x, sets_y, sets_z,
                           pidx, bxf);
    }   // openmp

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z,
                       bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);

    if (parms->debug) {
        fclose (fp);
    }

    mse_score = mse_score / num_vox;

    report_score ("MI", bxf, bst, num_vox, plm_timer_report (&timer));
}
#endif



/* B-Spline Registration using Mutual Information
 * Implementation E (D is still faster)
 *   -- Histograms are OpenMP accelerated
 *      (only good on i7 core? really bad otherwise it seems...)
 *   -- Uses OpenMP for Cost & dc_dv computation
 *   -- Uses methods introduced in bspline_score_g_mse
 *        to compute dc_dp more rapidly.
 */
#if (OPENMP_FOUND)
static void
bspline_score_e_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    int pidx;
    int num_vox;
    float num_vox_f;
    Timer timer;

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    static int it = 0;
    double mhis = 0.0f;      /* Moving histogram incomplete sum */
    double jhis = 0.0f;      /* Joint  histogram incomplete sum */
    char debug_fn[1024];
    FILE* fp;
    int i, j, zz;
	omp_lock_t *f_locks, *m_locks, *j_locks;

    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);


    if (parms->debug) {
        sprintf (debug_fn, "dump_mi_%02d.txt", it++);
        fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);
    num_vox = 0;

    /* -- OpenMP locks for histograms --------------------- */
    f_locks = (omp_lock_t*) malloc (mi_hist->fixed.bins * sizeof(omp_lock_t));
    m_locks = (omp_lock_t*) malloc (mi_hist->moving.bins * sizeof(omp_lock_t));
    j_locks = (omp_lock_t*) malloc (mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(omp_lock_t));

#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins; i++) {
        omp_init_lock(&f_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->moving.bins; i++) {
        omp_init_lock(&m_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins * mi_hist->moving.bins; i++) {
        omp_init_lock(&j_locks[i]);
    }
    /* ---------------------------------------------------- */

    /* PASS 1 - Accumulate histogram */
#pragma omp parallel for
    for (pidx=0; pidx < num_tiles; pidx++) {
        int rc;
        int fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        int mijk_f[3], mvf;      /* Floor */
        int mijk_r[3];           /* Round */
        int p[3];
        int q[3];
        float dxyz[3];
        int qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        for (q[2]=0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
            for (q[1]=0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
                for (q[0]=0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
                    
                    /* Construct coordinates into fixed image volume */
                    fijk[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0]*p[0] + q[0];
                    fijk[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1]*p[1] + q[1];
                    fijk[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2]*p[2] + q[2];
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];
                    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
                    fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];

                    /* Construct the linear index within tile space */
                    qidx = INDEX_OF (q, bxf->vox_per_rgn);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
                                                 li_1, li_2,
                                                 moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = INDEX_OF (fijk, fixed->dim);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = INDEX_OF (mijk_f, moving->dim);

                    /* Add to histogram */

                    bspline_mi_hist_add_pvi_8_omp (mi_hist, fixed, moving, 
                                                   fv, mvf, li_1, li_2,
                                                   f_locks, m_locks, j_locks);
#if defined (commentout)
#endif
                }
            }
        }   // tile
    }   // openmp

    /* Compute num_vox and find fullest fixed hist bin */
    for(i=0; i<mi_hist->fixed.bins; i++) {
        if (f_hist[i] > f_hist[mi_hist->fixed.big_bin]) {
            mi_hist->fixed.big_bin = i;
        }
        num_vox += f_hist[i];
    }

    /* Fill in the missing histogram bin */
    for(i=0; i<mi_hist->moving.bins; i++) {
        mhis += m_hist[i];
    }
    m_hist[mi_hist->moving.big_bin] = (double)num_vox - mhis;


    /* Look for the biggest moving histogram bin */
//    printf ("moving.big_bin [%i -> ", mi_hist->moving.big_bin);
    for(i=0; i<mi_hist->moving.bins; i++) {
        if (m_hist[i] > m_hist[mi_hist->moving.big_bin]) {
            mi_hist->moving.big_bin = i;
        }
    }
//    printf ("%i]\n", mi_hist->moving.big_bin);


    /* Fill in the missing jnt hist bin */
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            jhis += j_hist[j*mi_hist->moving.bins + i];
        }
    }
    j_hist[mi_hist->joint.big_bin] = (double)num_vox - jhis;

    
    /* Look for the biggest joint histogram bin */
//    printf ("joint.big_bin [%i -> ", mi_hist->joint.big_bin);
    for(j=0; j<mi_hist->fixed.bins; j++) {
        for(i=0; i<mi_hist->moving.bins; i++) {
            if (j_hist[j*mi_hist->moving.bins + i] > j_hist[mi_hist->joint.big_bin]) {
                mi_hist->joint.big_bin = j*mi_hist->moving.bins + i;
            }
        }
    }
//    printf ("%i]\n", mi_hist->joint.big_bin);
    


    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->score = mi_hist_score_omp (mi_hist, num_vox);
    num_vox_f = (float) num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    for (pidx=0; pidx < num_tiles; pidx++) {
        int rc;
        int fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        int mijk_f[3], mvf;      /* Floor */
        int mijk_r[3];           /* Round */
        int p[3];
        int q[3];
        float dxyz[3];
        int qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));

        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        for (q[2]=0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
            for (q[1]=0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
                for (q[0]=0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
                    
                    /* Construct coordinates into fixed image volume */
                    fijk[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0]*p[0] + q[0];
                    fijk[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1]*p[1] + q[1];
                    fijk[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2]*p[2] + q[2];
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];
                    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
                    fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];

                    /* Construct the linear index within tile space */
                    qidx = INDEX_OF (q, bxf->vox_per_rgn);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
                                                 li_1, li_2,
                                                 moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = INDEX_OF (fijk, fixed->dim);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = INDEX_OF (mijk_f, moving->dim);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                        fv, mvf, mijk, num_vox_f, li_1, li_2);

                    /* Update condensed tile sets */
                    bspline_update_sets (sets_x, sets_y, sets_z,
                                         qidx, dc_dv, bxf);
                }
            }
        }   // tile

        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
                           sets_x, sets_y, sets_z,
                           pidx, bxf);
    }   // openmp

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z,
                       bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);


#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins; i++) {
        omp_destroy_lock(&f_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->moving.bins; i++) {
        omp_destroy_lock(&m_locks[i]);
    }

#pragma omp parallel for
    for (i=0; i < mi_hist->fixed.bins * mi_hist->moving.bins; i++) {
        omp_destroy_lock(&j_locks[i]);
    }



    if (parms->debug) {
        fclose (fp);
    }

    mse_score = mse_score / num_vox;

    report_score ("MI", bxf, bst, num_vox, plm_timer_report (&timer));
}
#endif


/* B-Spline Registration using Mutual Information
 * Implementation D
 *   -- Uses OpenMP for Cost & dc_dv computation
 *   -- Uses methods introduced in bspline_score_g_mse
 *        to compute dc_dp more rapidly.
 */
static void
bspline_score_d_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
    BSPLINE_Score* ssd = &bst->ssd;
    BSPLINE_MI_Hist* mi_hist = &parms->mi_hist;
    int rijk[3];
    float diff;
    float* f_img = (float*) fixed->img;
    float* m_img = (float*) moving->img;
    int num_vox;
    float num_vox_f;
    int pidx;
    Timer timer;
    float m_val;

    int fijk[3], fv;
    float mijk[3];
    float fxyz[3];
    float mxyz[3];
    int mijk_f[3], mvf;      /* Floor */
    int mijk_r[3];           /* Round */
    int p[3];
    int q[3];
    float dxyz[3];
    int qidx;
    float li_1[3];           /* Fraction of interpolant in lower index */
    float li_2[3];           /* Fraction of interpolant in upper index */

    float mse_score = 0.0f;
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
    static int it = 0;
    char debug_fn[1024];
    FILE* fp;
    int zz;

    int num_tiles = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    size_t cond_size = 64*bxf->num_knots*sizeof(float);
    float* cond_x = (float*)malloc(cond_size);
    float* cond_y = (float*)malloc(cond_size);
    float* cond_z = (float*)malloc(cond_size);



    if (parms->debug) {
	sprintf (debug_fn, "dump_mi_%02d.txt", it++);
	fp = fopen (debug_fn, "w");
    }

    plm_timer_start (&timer);

    memset (ssd->grad, 0, bxf->num_coeff * sizeof(float));
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
    memset(cond_x, 0, cond_size);
    memset(cond_y, 0, cond_size);
    memset(cond_z, 0, cond_size);
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
            bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

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
            // NOTE: Not used by MI PVI8
            BSPLINE_LI_VALUE (m_val, 
                li_1[0], li_2[0],
                li_1[1], li_2[1],
                li_1[2], li_2[2],
                mvf, m_img, moving);

#if defined (commentout)
            /* LINEAR INTERPOLATION */
            bspline_mi_hist_add (mi_hist, f_img[fv], m_val, 1.0);
#endif

            /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
            bspline_mi_hist_add_pvi_8 (mi_hist, fixed, moving, 
                fv, mvf, li_1, li_2);

#if defined (commentout)
            /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
            bspline_mi_hist_add_pvi_6 (mi_hist, fixed, moving, 
                fv, mvf, mijk);
#endif

            /* Compute intensity difference */
            diff = m_val - f_img[fv];
            mse_score += diff * diff;
            num_vox ++;
        }
    }
    }


    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
            tmp += j_hist[zz];
        }
        printf ("j_hist total: %f\n", tmp);
    }

    /* Compute score */
    ssd->score = mi_hist_score_omp (mi_hist, num_vox);
    num_vox_f = (float) num_vox;

    /* PASS 2 - Compute Gradient (Parallel across tiles) */
#pragma omp parallel for
    for (pidx=0; pidx < num_tiles; pidx++) {
        int rc;
        int fijk[3], fv;
        float mijk[3];
        float fxyz[3];
        float mxyz[3];
        int mijk_f[3], mvf;      /* Floor */
        int mijk_r[3];           /* Round */
        int p[3];
        int q[3];
        float dxyz[3];
        int qidx;
        float li_1[3];           /* Fraction of interpolant in lower index */
        float li_2[3];           /* Fraction of interpolant in upper index */
        float dc_dv[3];
        float sets_x[64];
        float sets_y[64];
        float sets_z[64];

        memset(sets_x, 0, 64*sizeof(float));
        memset(sets_y, 0, 64*sizeof(float));
        memset(sets_z, 0, 64*sizeof(float));


        /* Get tile indices from linear index */
        COORDS_FROM_INDEX (p, pidx, bxf->rdims);

        /* Serial through the voxels in a tile */
        for (q[2]=0; q[2] < bxf->vox_per_rgn[2]; q[2]++) {
            for (q[1]=0; q[1] < bxf->vox_per_rgn[1]; q[1]++) {
                for (q[0]=0; q[0] < bxf->vox_per_rgn[0]; q[0]++) {
                    
                    /* Construct coordinates into fixed image volume */
                    fijk[0] = bxf->roi_offset[0] + bxf->vox_per_rgn[0]*p[0] + q[0];
                    fijk[1] = bxf->roi_offset[1] + bxf->vox_per_rgn[1]*p[1] + q[1];
                    fijk[2] = bxf->roi_offset[2] + bxf->vox_per_rgn[2]*p[2] + q[2];
                    
                    /* Check to make sure the indices are valid (inside volume) */
                    if (fijk[0] >= bxf->roi_offset[0] + bxf->roi_dim[0]) { continue; }
                    if (fijk[1] >= bxf->roi_offset[1] + bxf->roi_dim[1]) { continue; }
                    if (fijk[2] >= bxf->roi_offset[2] + bxf->roi_dim[2]) { continue; }

                    /* Compute space coordinates of fixed image voxel */
                    fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];
                    fxyz[1] = bxf->img_origin[1] + bxf->img_spacing[1] * fijk[1];
                    fxyz[2] = bxf->img_origin[2] + bxf->img_spacing[2] * fijk[2];

                    /* Construct the linear index within tile space */
                    qidx = INDEX_OF (q, bxf->vox_per_rgn);

                    /* Compute deformation vector (dxyz) for voxel */
                    bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

                    /* Find correspondence in moving image */
                    rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

                    /* If voxel is not inside moving image */
                    if (!rc) continue;

                    /* Get tri-linear interpolation fractions */
                    CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
                                                 li_1, li_2,
                                                 moving);
                    
                    /* Constrcut the fixed image linear index within volume space */
                    fv = INDEX_OF (fijk, fixed->dim);

                    /* Find linear index the corner voxel used to identifiy the
                     * neighborhood of the moving image voxels corresponding
                     * to the current fixed image voxel */
                    mvf = INDEX_OF (mijk_f, moving->dim);

                    /* Compute dc_dv */
                    bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                        fv, mvf, mijk, num_vox_f, li_1, li_2);

                    /* Update condensed tile sets */
                    bspline_update_sets (sets_x, sets_y, sets_z,
                                         qidx, dc_dv, bxf);
                }
            }
        }   // tile

        /* We now have a tile of condensed dc_dv values (64 values).
         * Let's put each one in the proper slot within the control
         * point bin its belogs to */
        bspline_sort_sets (cond_x, cond_y, cond_z,
                           sets_x, sets_y, sets_z,
                           pidx, bxf);
    }   // openmp

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z,
                       bxf, ssd);

    free (cond_x);
    free (cond_y);
    free (cond_z);


    if (parms->debug) {
        fclose (fp);
    }

    mse_score = mse_score / num_vox;

    report_score ("MI", bxf, bst, num_vox, plm_timer_report (&timer));
}



/* Mutual information version of implementation "C" */
static void
bspline_score_c_mi (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform *bxf, 
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
    double* f_hist = mi_hist->f_hist;
    double* m_hist = mi_hist->m_hist;
    double* j_hist = mi_hist->j_hist;
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
    memset (f_hist, 0, mi_hist->fixed.bins * sizeof(double));
    memset (m_hist, 0, mi_hist->moving.bins * sizeof(double));
    memset (j_hist, 0, mi_hist->fixed.bins * mi_hist->moving.bins * sizeof(double));
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
            bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

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
            // NOTE: Not used by MI PVI8
            BSPLINE_LI_VALUE (m_val, 
                li_1[0], li_2[0],
                li_1[1], li_2[1],
                li_1[2], li_2[2],
                mvf, m_img, moving);

#if defined (commentout)
            /* LINEAR INTERPOLATION */
            bspline_mi_hist_add (mi_hist, f_img[fv], m_val, 1.0);
#endif

            /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
            bspline_mi_hist_add_pvi_8 (mi_hist, fixed, moving, 
                fv, mvf, li_1, li_2);

#if defined (commentout)
            /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
            bspline_mi_hist_add_pvi_6 (mi_hist, fixed, moving, 
                fv, mvf, mijk);
#endif

            /* Compute intensity difference */
            diff = m_val - f_img[fv];
            mse_score += diff * diff;
            num_vox ++;
        }
    }
    }


    /* Draw histogram images if user wants them */
    if (parms->xpm_hist_dump) {
        dump_xpm_hist (mi_hist, parms->xpm_hist_dump, bst->it);
    }

    /* Display histrogram stats in debug mode */
    if (parms->debug) {
        double tmp;
        for (zz=0,tmp=0; zz < mi_hist->fixed.bins; zz++) {
            tmp += f_hist[zz];
        }
        printf ("f_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins; zz++) {
            tmp += m_hist[zz];
        }
        printf ("m_hist total: %f\n", tmp);

        for (zz=0,tmp=0; zz < mi_hist->moving.bins * mi_hist->fixed.bins; zz++) {
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
            int rc;

            p[0] = rijk[0] / bxf->vox_per_rgn[0];
            q[0] = rijk[0] % bxf->vox_per_rgn[0];
            fxyz[0] = bxf->img_origin[0] + bxf->img_spacing[0] * fijk[0];

            /* Get B-spline deformation vector */
            pidx = INDEX_OF (p, bxf->rdims);
            qidx = INDEX_OF (q, bxf->vox_per_rgn);
            bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

            /* Find linear index of fixed image voxel */
            fv = INDEX_OF (fijk, fixed->dim);

            /* Find correspondence in moving image */
            rc = bspline_find_correspondence (mxyz, mijk, fxyz, dxyz, moving);

            /* If voxel is not inside moving image */
            if (!rc) continue;

            /* LINEAR INTERPOLATION - (not implemented) */

            /* PARTIAL VALUE INTERPOLATION - 8 neighborhood */
            CLAMP_LINEAR_INTERPOLATE_3D (mijk, mijk_f, mijk_r, 
                li_1, li_2, moving);

            /* Find linear index of fixed image voxel */
            fv = INDEX_OF (fijk, fixed->dim);

            /* Find linear index of "corner voxel" in moving image */
            mvf = INDEX_OF (mijk_f, moving->dim);

            bspline_mi_pvi_8_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                fv, mvf, mijk, num_vox_f, li_1, li_2);

#if defined (commentout)
            /* PARTIAL VALUE INTERPOLATION - 6 neighborhood */
            bspline_mi_pvi_6_dc_dv (dc_dv, mi_hist, bst, fixed, moving, 
                fv, mvf, mijk, num_vox_f);
#endif

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
    Bspline_xform* bxf, /* Bspline transform coefficients */
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
		bspline_interp_pix_b (dxyz, bxf, pidx, qidx);

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
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
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

	memset(sets_x, 0, 64*sizeof(float));
	memset(sets_y, 0, 64*sizeof(float));
	memset(sets_z, 0, 64*sizeof(float));

	// Get tile coordinates from index
	COORDS_FROM_INDEX (crds_tile, idx_tile, bxf->rdims); 

	// Serial through voxels in tile
	for (crds_local[2] = 0; crds_local[2] < bxf->vox_per_rgn[2]; crds_local[2]++) {
	    for (crds_local[1] = 0; crds_local[1] < bxf->vox_per_rgn[1]; crds_local[1]++) {
		for (crds_local[0] = 0; crds_local[0] < bxf->vox_per_rgn[0]; crds_local[0]++) {
					
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
		    bspline_interp_pix_b (dxyz, bxf, idx_tile, idx_local);

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
		    diff = m_val - f_img[idx_fixed];

		    // Store the score!
		    score_tile += diff * diff;
		    num_vox++;

		    // Compute dc_dv
		    dc_dv[0] = diff * m_grad[3 * idx_moving_round + 0];
		    dc_dv[1] = diff * m_grad[3 * idx_moving_round + 1];
		    dc_dv[2] = diff * m_grad[3 * idx_moving_round + 2];

            /* Generate condensed tile */
            bspline_update_sets (sets_x, sets_y, sets_z,
                                 idx_local, dc_dv, bxf);
		}
	    }
	}
		
	// The tile is now condensed.  Now we will put it in the
	// proper slot within the control point bin that it belong to.
    bspline_sort_sets (cond_x, cond_y, cond_z,
                       sets_x, sets_y, sets_z,
                       idx_tile, bxf);


    }

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

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
//    (That does not require SSE)
//
// AUTHOR: James A. Shackleford
// DATE: 11.22.2009
////////////////////////////////////////////////////////////////////////////////
void
bspline_score_g_mse (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
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

	memset(sets_x, 0, 64*sizeof(float));
	memset(sets_y, 0, 64*sizeof(float));
	memset(sets_z, 0, 64*sizeof(float));

	// Get tile coordinates from index
	COORDS_FROM_INDEX (crds_tile, idx_tile, bxf->rdims); 

	// Serial through voxels in tile
	for (crds_local[2] = 0; crds_local[2] < bxf->vox_per_rgn[2]; crds_local[2]++) {
	    for (crds_local[1] = 0; crds_local[1] < bxf->vox_per_rgn[1]; crds_local[1]++) {
		for (crds_local[0] = 0; crds_local[0] < bxf->vox_per_rgn[0]; crds_local[0]++) {
					
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
		    bspline_interp_pix_b (dxyz, bxf, idx_tile, idx_local);

		    // Calc. moving image coordinate from the deformation vector
		    rc = bspline_find_correspondence (phys_moving,
			crds_moving,
			phys_fixed,
			dxyz,
			moving);

		    // Return code is 0 if voxel is pushed outside of 
		    // moving image
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
		    diff = m_val - f_img[idx_fixed];

		    // Store the score!
		    score_tile += diff * diff;
		    num_vox++;

		    // Compute dc_dv
		    dc_dv[0] = diff * m_grad[3 * idx_moving_round + 0];
		    dc_dv[1] = diff * m_grad[3 * idx_moving_round + 1];
		    dc_dv[2] = diff * m_grad[3 * idx_moving_round + 2];

            /* Generate condensed tile */
            bspline_update_sets (sets_x, sets_y, sets_z,
                                 idx_local, dc_dv, bxf);
		}
	    }
	}

	// The tile is now condensed.  Now we will put it in the
	// proper slot within the control point bin that it belong to.
    bspline_sort_sets (cond_x, cond_y, cond_z,
                       sets_x, sets_y, sets_z,
                       idx_tile, bxf);
    }

    /* Now we have a ton of bins and each bin's 64 slots are full.
     * Let's sum each bin's 64 slots.  The result with be dc_dp. */
    bspline_make_grad (cond_x, cond_y, cond_z, bxf, ssd);

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
bspline_score (Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad)
{
#if (CUDA_FOUND)
    if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MSE)) {

    /* Be sure we loaded the CUDA plugin */
	if (!delayload_cuda ()) { exit (0); }
    LOAD_LIBRARY (libplmcuda);
    LOAD_SYMBOL (CUDA_bspline_mse_j, libplmcuda);

	switch (parms->implementation) {
#if defined (commentout)
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
#endif
	case 'j':
	    CUDA_bspline_mse_j (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	default:
	    CUDA_bspline_mse_j (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	}

    UNLOAD_LIBRARY (libplmcuda);
	return;

    } else if ((parms->threading == BTHR_CUDA) && (parms->metric == BMET_MI)) {

    /* Be sure we loaded the CUDA plugin */
	if (!delayload_cuda ()) { exit (0); }
    LOAD_LIBRARY (libplmcuda);
    LOAD_SYMBOL (CUDA_bspline_mi_a, libplmcuda);

	switch (parms->implementation) {
	case 'a':
        CUDA_bspline_mi_a (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	default: 
        CUDA_bspline_mi_a (parms, bst, bxf, fixed, moving, moving_grad, bst->dev_ptrs);
	    break;
	}

    UNLOAD_LIBRARY (libplmcuda);
    }
#endif

    if (parms->metric == BMET_MSE) {
	switch (parms->implementation) {
#if defined (commentout)
	case 'a':
	    bspline_score_a_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'b':
	    bspline_score_b_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
#endif
	case 'c':
	    bspline_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
#if defined (commentout)
	case 'd':
	    bspline_score_d_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'e':
	    bspline_score_e_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'f':
	    bspline_score_f_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
#endif
	case 'g':
	    bspline_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'h':
	    bspline_score_h_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	default:
	    bspline_score_g_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	}
    }

    if ((parms->threading == BTHR_CPU) && (parms->metric == BMET_MI)) {
	switch (parms->implementation) {
	case 'c':
	    bspline_score_c_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
#if (OPENMP_FOUND)
	case 'd':
	    bspline_score_d_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
	case 'e':
	    bspline_score_e_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
    case 'f':
	    bspline_score_f_mi (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
#endif
	default:
#if (OPENMP_FOUND)
	    bspline_score_d_mi (parms, bst, bxf, fixed, moving, moving_grad);
#else
	    printf ("OpenMP not available. Defaulting to single core...\n");
	    bspline_score_c_mi (parms, bst, bxf, fixed, moving, moving_grad);
#endif
	    break;
	}
    }

    /* Add vector field score/gradient to image score/gradient */
    if (parms->young_modulus) {
	printf ("comuting regularization\n");
	bspline_regularize_score (parms, bst, bxf, fixed, moving);
    }

#if defined (commentout)
    /* Add landmark score/gradient to image score/gradient */
    if (parms->landmarks) {
	printf ("comuting landmarks\n");
	bspline_landmarks_score (parms, bst, bxf, fixed, moving);
    }
#endif
}

void
bspline_run_optimization (
    Bspline_xform* bxf, 
    Bspline_state **bst_in, 
    Bspline_parms *parms, 
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

    if (parms->young_modulus !=0) {
	bspline_xform_create_qlut_grad (bxf, bxf->img_spacing, bxf->vox_per_rgn);
    }

    /* Do the optimization */
    bspline_optimize (bxf, bst, parms, fixed, moving, moving_grad);

    if (parms->young_modulus !=0) {
	bspline_xform_free_qlut_grad (bxf);
    }

    if (bst_in) {
	*bst_in = bst;
    } else {
	bspline_state_destroy (bst, parms, fixed, moving, moving_grad);
    }
}
