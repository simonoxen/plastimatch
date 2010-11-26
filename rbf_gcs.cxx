/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vnl/vnl_matrix_fixed.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/vnl_vector_fixed.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/vnl_sample.h"

#include "landmark_warp.h"
#include "logfile.h"
#include "print_and_exit.h"
#include "rbf_gcs.h"
#include "vf.h"
#include "volume.h"

#define BUFLEN 1024
#define MOTION_TOL 0.01
#define DIST_MULTIPLIER 5.0
#define DIST2_MULTIPLIER (DIST_MULTIPLIER * DIST_MULTIPLIER)

#define INDEX_OF(ijk, dim) \
    (((ijk[2] * dim[1] + ijk[1]) * dim[0]) + ijk[0])

static void tps_xform_debug (Tps_xform *tps);

void
rbf_gcs_warp (Landmark_warp *lw)
{
    Tps_xform *tps;
    Volume *moving;
    Volume *vf_out = 0;
    Volume *warped_out = 0;

    printf ("Gonna convert pointsets\n");
    pointset_debug (lw->m_fixed_landmarks);

    /* Convert Landmark_warp to Tps_xform */
    /* GCS FIX: this function should use lw directly */
    tps = tps_xform_alloc ();

    lw->m_pih.get_origin (tps->img_origin);
    lw->m_pih.get_spacing (tps->img_spacing);
    lw->m_pih.get_dim (tps->img_dim);

    /* GCS FIX: Assume same number of points in both pointsets */
    tps->num_tps_nodes = lw->m_fixed_landmarks->num_points;
    tps->tps_nodes = (struct tps_node*) malloc (
	tps->num_tps_nodes * sizeof (Tps_node));
    for (int i = 0; i < tps->num_tps_nodes; i++) {
	Tps_node *curr_node = &tps->tps_nodes[i];
	curr_node->src[0] = lw->m_fixed_landmarks->points[i*3+0];
	curr_node->src[1] = lw->m_fixed_landmarks->points[i*3+1];
	curr_node->src[2] = lw->m_fixed_landmarks->points[i*3+2];
	curr_node->tgt[0] = lw->m_moving_landmarks->points[i*3+0];
	curr_node->tgt[1] = lw->m_moving_landmarks->points[i*3+1];
	curr_node->tgt[2] = lw->m_moving_landmarks->points[i*3+2];
	curr_node->alpha = tps_default_alpha (
	    curr_node->src, curr_node->tgt);

	/* Compute initial weights, based on distance and alpha */
	curr_node->wxyz[2] = curr_node->tgt[2] - curr_node->src[2];
	curr_node->wxyz[1] = curr_node->tgt[1] - curr_node->src[1];
	curr_node->wxyz[0] = curr_node->tgt[0] - curr_node->src[0];

    }

    tps_xform_debug (tps);

    printf ("Converting volume to float\n");
    moving = lw->m_input_img->gpuit_float ();

    printf ("Creating output vf\n");
    vf_out = volume_create (
	tps->img_dim, 
	tps->img_origin, 
	tps->img_spacing, 
	PT_VF_FLOAT_INTERLEAVED, 
	0, 0);

    printf ("Creating output vol\n");
    warped_out = volume_create (
	tps->img_dim, 
	tps->img_origin, 
	tps->img_spacing, 
	PT_FLOAT, 
	0, 0);
	
    printf ("Calling tps_warp... (default = %f)\n", lw->default_val);
    tps_warp (warped_out, vf_out, tps, moving, 1, lw->default_val);
    printf ("done!\n");

    /* Copy outputs to lw structure */
    lw->m_vf = new Xform;
    lw->m_vf->set_gpuit_vf (vf_out);
    lw->m_warped_img = new Plm_image;
    lw->m_warped_img->set_gpuit (warped_out);

    tps_xform_destroy (tps);
    printf ("Finished.\n");
}

Tps_xform*
tps_xform_alloc (void)
{
    Tps_xform *tps;

    tps = (Tps_xform*) malloc (sizeof (Tps_xform));
    if (!tps) {
	print_and_exit ("Out of memory\n");
    }
    memset (tps, 0, sizeof (Tps_xform));
    return tps;
}

Tps_xform*
tps_xform_load (char* fn)
{
    FILE *fp;
    Tps_xform *tps;
    char buf[1024];
    int rc;
    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    int d;

    tps = tps_xform_alloc ();

    /* If file doesn't exist, then no tps for this phase */
    fp = fopen (fn, "r");
    if (!fp) return tps;

    /* Skip first line */
    fgets (buf, 1024, fp);

    /* Read header */
    rc = fscanf (fp, "img_origin = %f %f %f\n", 
	&img_origin[0], &img_origin[1], &img_origin[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_origin): %s\n", fn);
	return tps;
    }
    rc = fscanf (fp, "img_spacing = %f %f %f\n", 
	&img_spacing[0], &img_spacing[1], &img_spacing[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_spacing): %s\n", fn);
	return tps;
    }
    rc = fscanf (fp, "img_dim = %d %d %d\n", 
	&img_dim[0], &img_dim[1], &img_dim[2]);
    if (rc != 3) {
	logfile_printf ("Error parsing input xform (img_dim): %s\n", fn);
	return tps;
    }

    for (d = 0; d < 3; d++) {
	tps->img_spacing[d] = img_spacing[d];
	tps->img_origin[d] = img_origin[d];
	tps->img_dim[d] = img_dim[d];
    }

    /* Read control points */
    while (fgets (buf, 1024, fp)) {
	int rc;
	Tps_node *curr_node;
	tps->tps_nodes = (struct tps_node*) 
	    realloc ((void*) (tps->tps_nodes), 
		(tps->num_tps_nodes + 1) * sizeof (Tps_node));
	if (!(tps->tps_nodes)) {
	    print_and_exit ("Error allocating memory");
	}
	curr_node = &tps->tps_nodes[tps->num_tps_nodes];
	rc = sscanf (buf, "%g %g %g %g %g %g %g", 
	    &curr_node->src[0],
	    &curr_node->src[1],
	    &curr_node->src[2],
	    &curr_node->tgt[0],
	    &curr_node->tgt[1],
	    &curr_node->tgt[2],
	    &curr_node->alpha
	);
	if (rc != 7) {
	    print_and_exit ("Ill-formed input file: %s\n", fn);
	}

	/* Compute initial weights, based on distance and alpha */
	curr_node->wxyz[2] = curr_node->tgt[2] - curr_node->src[2];
	curr_node->wxyz[1] = curr_node->tgt[1] - curr_node->src[1];
	curr_node->wxyz[0] = curr_node->tgt[0] - curr_node->src[0];

	tps->num_tps_nodes++;
    }
    fclose (fp);

    tps_xform_reset_alpha_values (tps);

    return tps;
}

/* GCS FIX: Current best version in plastimatch-slicer-tps.cxx */
#if defined (commentout)
void
tps_xform_save (Tps_xform *tps, char *fn)
{
    int i;
    FILE *fp = fopen (fn, "w");
    if (!fp) {
	print_and_exit ("Couldn't open file \"%s\" for write\n", fn);
    }

    for (i = 0; i < tps->num_tps_nodes; i++) {
	float dist;
	float *src = tps->tps_nodes[i].src;
	float *tgt = tps->tps_nodes[i].tgt;
	dist = sqrt (((src[0] - tgt[0]) * (src[0] - tgt[0]))
	    + ((src[1] - tgt[1]) * (src[1] - tgt[1]))
	    + ((src[2] - tgt[2]) * (src[2] - tgt[2])));

	fprintf (fp, "%g %g %g %g %g %g %g\n", 
	    src[0],
	    src[1],
	    src[2],
	    tgt[0]-src[0],
	    tgt[1]-src[1],
	    tgt[2]-src[2],
	    DIST2_MULTIPLIER * dist
	);
    }
    fclose (fp);
}
#endif

void
tps_xform_destroy (Tps_xform *tps)
{
    free (tps->tps_nodes);
    free (tps);
}

static void
tps_xform_debug (Tps_xform *tps)
{
    int i;
    printf ("TPS XFORM (%d nodes)\n", tps->num_tps_nodes);
    for (i = 0; i < tps->num_tps_nodes; i++) {
	Tps_node *curr_node = &tps->tps_nodes[i];
	printf ("src %5.1f %5.1f %5.1f tgt %5.1f %5.1f %5.1f "
	    "wxyz %5.1f %5.1f %5.1f a %5.1f\n", 
	    curr_node->src[0],
	    curr_node->src[1],
	    curr_node->src[2],
	    curr_node->tgt[0],
	    curr_node->tgt[1],
	    curr_node->tgt[2],
	    curr_node->wxyz[0],
	    curr_node->wxyz[1],
	    curr_node->wxyz[2],
	    curr_node->alpha
	);
    }
}

void
tps_xform_reset_alpha_values (Tps_xform *tps)
{
    int i;
    for (i = 0; i < tps->num_tps_nodes; i++) {
	Tps_node *curr_node = &tps->tps_nodes[i];
	curr_node->alpha = tps_default_alpha (curr_node->src, curr_node->tgt);
    }
}

float
tps_default_alpha (float src[3], float tgt[3])
{
    float dist2;

    dist2 = ((src[0] - tgt[0]) * (src[0] - tgt[0]))
	+ ((src[1] - tgt[1]) * (src[1] - tgt[1]))
	+ ((src[2] - tgt[2]) * (src[2] - tgt[2]));
    return DIST2_MULTIPLIER * dist2;
}

static double
tps_compute_influence (Tps_node *tpsn, float pos[3])
{
    float dist2 = 
	((pos[0] - tpsn->src[0]) * (pos[0] - tpsn->src[0])) +
	((pos[1] - tpsn->src[1]) * (pos[1] - tpsn->src[1])) +
	((pos[2] - tpsn->src[2]) * (pos[2] - tpsn->src[2]));
    dist2 = dist2 / tpsn->alpha;
    if (dist2 < 1.0) {
	float weight = (1 - dist2);
	return (double) weight;
    }
    return 0.0;
}

static void
tps_update_point (
    float vf[3],       /* Output: displacement to update */
    Tps_node* tpsn,    /* Input: the tps control point */
    float pos[3])      /* Input: location of voxel to update */
{
    int d;

    float dist2 = 
	((pos[0] - tpsn->src[0]) * (pos[0] - tpsn->src[0])) +
	((pos[1] - tpsn->src[1]) * (pos[1] - tpsn->src[1])) +
	((pos[2] - tpsn->src[2]) * (pos[2] - tpsn->src[2]));
    dist2 = dist2 / tpsn->alpha;
    if (dist2 < 1.0) {
	float weight = (1 - dist2);
	for (d = 0; d < 3; d++) {
	    vf[d] += weight * tpsn->wxyz[d];
	}
    }
}

void
tps_solve (Tps_xform *tps)
{
    typedef vnl_matrix <double> Vnl_matrix;
    typedef vnl_svd <double> SVDSolverType;

    int cpi1, cpi2;
    Vnl_matrix A, b;
    
    A.set_size (3 * tps->num_tps_nodes, 3 * tps->num_tps_nodes);
    A.set_identity ();

    b.set_size (3 * tps->num_tps_nodes, 1);
    b.fill (0.0);

    tps_xform_debug (tps);

    for (cpi1 = 0; cpi1 < tps->num_tps_nodes; cpi1++) {
	Tps_node *curr_node = &tps->tps_nodes[cpi1];
	for (cpi2 = 0; cpi2 < tps->num_tps_nodes; cpi2++) {
	    double w;
	    w = tps_compute_influence (curr_node, tps->tps_nodes[cpi2].src);

	    for (int d = 0; d < 3; d++) {
		A (3 * cpi1 + d, 3 * cpi2 + d) = w;
	    }
	}
    }

    for (cpi1 = 0; cpi1 < tps->num_tps_nodes; cpi1++) {
	Tps_node *curr_node = &tps->tps_nodes[cpi1];
	for (int d = 0; d < 3; d++) {
	    b (3 * cpi1 + d, 0) = curr_node->tgt[d] - curr_node->src[d];
	}
    }

    A.print (std::cout);
    b.print (std::cout);

    SVDSolverType svd (A, 1e-8);
    Vnl_matrix x = svd.solve (b);

    x.print (std::cout);

    for (cpi1 = 0; cpi1 < tps->num_tps_nodes; cpi1++) {
	Tps_node *curr_node = &tps->tps_nodes[cpi1];
	for (int d = 0; d < 3; d++) {
	    curr_node->wxyz[d] = x (3 * cpi1 + d, 0);
	}
    }
}

void
tps_hack (Tps_xform *tps)
{
    int max_its = tps->num_tps_nodes;
    int it, cpi1, cpi2;
    float *wxyz_scratch;
    wxyz_scratch = (float*) malloc (tps->num_tps_nodes * 3 * sizeof (float));

    for (it = 0; it < max_its; it++) {

	/* Loop through pairs of control points, accumulate deformation */
	for (cpi1 = 0; cpi1 < tps->num_tps_nodes; cpi1++) {
	    wxyz_scratch[3 * cpi1 + 0] = 0.f;
	    wxyz_scratch[3 * cpi1 + 1] = 0.f;
	    wxyz_scratch[3 * cpi1 + 2] = 0.f;
	    for (cpi2 = 0; cpi2 < tps->num_tps_nodes; cpi2++) {
		tps_update_point (&wxyz_scratch[3 * cpi1],
		    &tps->tps_nodes[cpi2],
		    tps->tps_nodes[cpi1].src);
	    }
	}

	/* Update wxyz to improve approximate at control points */
	for (cpi1 = 0; cpi1 < tps->num_tps_nodes; cpi1++) {
	    Tps_node *curr_node = &tps->tps_nodes[cpi1];
	    printf ("err[%d] = ", cpi1);
	    for (int d = 0; d < 3; d++) {
		float err = wxyz_scratch[3 * cpi1 + d] 
		    - (curr_node->tgt[d] - curr_node->src[d]);
		printf ("%g ", err);
		curr_node->wxyz[d] -= err;
	    }
	    printf ("\n");
	}
    }

    free (wxyz_scratch);
}

void
tps_warp (
    Volume *vout,       /* Output image (sized and allocated) */
    Volume *vf_out,     /* Output vf (sized and allocated, can be null) */
    Tps_xform *tps,     /* TPS control points */
    Volume *moving,     /* Input image */
    int linear_interp,  /* 1 = trilinear, 0 = nearest neighbors */
    float default_val   /* Fill in this value outside of image */
)
{
    int d;
    int vidx;

    int cpi;
    int fijk[3], fidx;       /* Indices within fixed image (vox) */
    float fxyz[3];           /* Position within fixed image (mm) */
    Volume *vf;
    float *vf_img;

    /* A few sanity checks */
    if (vout && vout->pix_type != PT_FLOAT) {
	fprintf (stderr, "Error: tps_warp pix type mismatch\n");
	return;
    }
    if (vf_out && vf_out->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Error: tps_warp requires interleaved vf\n");
	return;
    }

    /* Set defaults */
    if (vout) {
	float* vout_img = (float*) vout->img;
	for (vidx = 0; vidx < vout->npix; vidx++) {
	    vout_img[vidx] = default_val;
	}
    }
    if (vf_out) {
	vf = vf_out;
	memset (vf->img, 0, vf->pix_size * vf_out->npix);
    } else {
	vf = volume_create (
	    tps->img_dim,
	    tps->img_origin,
	    tps->img_spacing,
	    PT_VF_FLOAT_INTERLEAVED,
	    0, 0);
    }
    vf_img = (float*) vf->img;

    /* Hack */
    //tps_hack (tps);

    /* Solve */
    tps_solve (tps);
	
    /* Loop through control points, and construct the vector field */
    for (cpi = 0; cpi < tps->num_tps_nodes; cpi++) {
	Tps_node *curr_node = &tps->tps_nodes[cpi];
	long roi_offset[3];
	long roi_size[3];

	/* Compute "region of influence" for current control point. 
	   The region of interest is the set of voxels that the control 
	   point influences, clipped against the entire volume. */
	for (d = 0; d < 3; d++) {
	    float rmin, rmax;
	    long rmini, rmaxi;

	    rmin = curr_node->src[d] - curr_node->alpha;
	    rmax = curr_node->src[d] + curr_node->alpha;
	    rmini = (long) floor (rmin - tps->img_origin[d] 
		/ tps->img_spacing[d]);
	    rmaxi = (long) ceil (rmax - tps->img_origin[d] 
		/ tps->img_spacing[d]);

	    if (rmini < 0) rmini = 0;
	    if (rmaxi >= moving->dim[d]) rmaxi = moving->dim[d] - 1;

	    roi_offset[d] = rmini;
	    roi_size[d] = rmaxi - rmini + 1;
	}

	printf (
	    "Region[%d] offset = (%ld %ld %ld), size = (%ld %ld %ld)"
	    " alpha = %g\n",
	    cpi, roi_offset[0], roi_offset[1], roi_offset[2], 
	    roi_size[0], roi_size[1], roi_size[2],
	    curr_node->alpha
	);

	/* Loop through ROI */
	for (fijk[2] = roi_offset[2]; fijk[2] < roi_offset[2] + roi_size[2] - 1; fijk[2]++) {
	    fxyz[2] = tps->img_origin[2] + tps->img_spacing[2] * fijk[2];
	    for (fijk[1] = roi_offset[1]; fijk[1] < roi_offset[1] + roi_size[1] - 1; fijk[1]++) {
		fxyz[1] = tps->img_origin[1] + tps->img_spacing[1] * fijk[1];
		for (fijk[0] = roi_offset[0]; fijk[0] < roi_offset[0] + roi_size[0] - 1; fijk[0]++) {

		    /* Update vf at this voxel, for this node */
		    fxyz[0] = tps->img_origin[0] 
			+ tps->img_spacing[0] * fijk[0];
		    fidx = INDEX_OF (fijk, tps->img_dim);
		    tps_update_point (&vf_img[3*fidx], curr_node, fxyz);
		}
	    }
	}
    }

    /* Warp the image */
    /* GCS FIX: Does not implement linear interpolation */
    if (vout) {
	vf_warp (vout, moving, vf);
    }

    if (!vf_out) {
	volume_destroy (vf);
    }
}
