/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
    Proposed variable naming guide:
	Fixed image voxel                (f[3]), fidx <currently (fi,fj,fk),fv>
	Moving image voxel               (m[3]), midx < ditto >
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
#ifndef _WIN32
#include <dlfcn.h>
#endif
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif

#include "bspline.h"
#include "bspline_mi.h"
#include "bspline_mse.h"
#if (CUDA_FOUND)
#include "bspline_cuda.h"
#endif
#include "bspline_regularize.h"
#include "bspline_landmarks.h"
#include "bspline_optimize.h"
#include "bspline_optimize_lbfgsb.h"
#include "bspline_opts.h"
#include "delayload.h"
#include "interpolate.h"
#include "logfile.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_path.h"
#include "plm_timer.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_macros.h"
#include "xpm.h"

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
write_bxf (const char* filename, Bspline_xform* bxf)
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

/* This function uses the B-Spline coefficients to transform a point.  
   The point need not lie exactly on a voxel, so we do not use the 
   lookup table. */
void
bspline_transform_point (
    float point_out[3], /* Output coordinate of point */
    Bspline_xform* bxf, /* Bspline transform coefficients */
    float point_in[3],  /* Input coordinate of point */
    int linear_interp   /* 1 = trilinear, 0 = nearest neighbors */
)
{
    int d, i, j, k;
    int p[3];                    /* Index of tile */
    float q[3];                  /* Fractional offset within tile */
    float q_mini[3][4];          /* "miniature" q-lut, just for this point */

    /* Default value is untransformed point */
    for (d = 0; d < 3; d++) {
	point_out[d] = point_in[d];
    }

    /* Compute tile and offset within tile */
    for (d = 0; d < 3; d++) {
	float img_ijk[3];         /* Voxel coordinate of point_in */
	img_ijk[d] = (point_in[d] - bxf->img_origin[d]) / bxf->img_spacing[d];
	p[d] = (int) floorf (
	    (img_ijk[d] - bxf->roi_offset[d]) / bxf->vox_per_rgn[d]);
	/* If point lies outside of B-spline domain, return point_in */
	if (p[d] < 0 || p[d] >= bxf->rdims[d]) {
	    printf ("Unwarped point, outside roi: %f %f %f\n", 
		point_out[0], point_out[1], point_out[2]);
	    return;
	}
	q[d] = ((img_ijk[d] - bxf->roi_offset[d])
	    - p[d] * bxf->vox_per_rgn[d]) / bxf->vox_per_rgn[d];
    }

#if defined (commentout)
    printf ("p = [%d, %d, %d], q = [%f, %f, %f]\n", 
	p[0], p[1], p[2], q[0], q[1], q[2]);
#endif

    /* Compute basis function values for this offset */
    for (d = 0; d < 3; d++) {
	float t3 = q[d]*q[d]*q[d];
	float t2 = q[d]*q[d];
	float t1 = q[d];
	q_mini[d][0] = (1.0/6.0) * (- 1.0 * t3 + 3.0 * t2 - 3.0 * t1 + 1.0);
	q_mini[d][1] = (1.0/6.0) * (+ 3.0 * t3 - 6.0 * t2            + 4.0);
	q_mini[d][2] = (1.0/6.0) * (- 3.0 * t3 + 3.0 * t2 + 3.0 * t1 + 1.0);
	q_mini[d][3] = (1.0/6.0) * (+ 1.0 * t3);
    }

    /* Compute displacement vector and add to point_out */
#if defined (commentout)
    printf ("---\n");
#endif
    for (k = 0; k < 4; k++) {
	for (j = 0; j < 4; j++) {
	    for (i = 0; i < 4; i++) {
		float ql;
		int cidx;

		cidx = (p[2] + k) * bxf->cdims[1] * bxf->cdims[0]
		    + (p[1] + j) * bxf->cdims[0]
		    + (p[0] + i);
		cidx = cidx * 3;
		ql = q_mini[0][i] * q_mini[1][j] * q_mini[2][k];

#if defined (commentout)
		printf ("(%f) + [%f] + [%f] = ", point_out[0],
		    ql, bxf->coeff[cidx+0]);
#endif

		point_out[0] += ql * bxf->coeff[cidx+0];
		point_out[1] += ql * bxf->coeff[cidx+1];
		point_out[2] += ql * bxf->coeff[cidx+2];

#if defined (commentout)
		printf (" = (%f)\n", point_out[0]);
#endif
	    }
	}
    }
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

		li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

		if (linear_interp) {
		    /* Find linear index of "corner voxel" in moving image */
		    mvf = INDEX_OF (mijk_f, moving->dim);

		    /* Compute moving image intensity using linear 
		       interpolation */
		    /* Macro is slightly faster than function */
		    LI_VALUE (m_val, 
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
	case 'j':
	    CUDA_bspline_mse_j (parms, bst, bxf, fixed, moving, 
		moving_grad, bst->dev_ptrs);
	    break;
	default:
	    CUDA_bspline_mse_j (parms, bst, bxf, fixed, moving, 
		moving_grad, bst->dev_ptrs);
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
	    CUDA_bspline_mi_a (parms, bst, bxf, fixed, moving, 
		moving_grad, bst->dev_ptrs);
	    break;
	default: 
	    CUDA_bspline_mi_a (parms, bst, bxf, fixed, moving, 
		moving_grad, bst->dev_ptrs);
	    break;
	}

	UNLOAD_LIBRARY (libplmcuda);
    }
#endif

    if (parms->metric == BMET_MSE) {
	switch (parms->implementation) {
	case 'c':
	    bspline_score_c_mse (parms, bst, bxf, fixed, moving, moving_grad);
	    break;
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
