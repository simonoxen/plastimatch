/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "logfile.h"
#include "print_and_exit.h"
#include "volume.h"
#include "tps.h"

#define BUFLEN 1024
#define MOTION_TOL 0.01
#define DIST_MULTIPLIER 5.0

char tgt_fn[256];
char src_fn[256];
char deform_dir[256];
char tps_fn[256];

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
	tps->num_tps_nodes++;
    }
    fclose (fp);
    return tps;
}

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
	    DIST_MULTIPLIER * dist
	);
    }
    fclose (fp);
}

void
tps_xform_free (Tps_xform *tps)
{
    free (tps->tps_nodes);
    free (tps);
}

#if defined (commentout)
void
tps_warp_point (
    float new_pos[3], 
    Tps_xform* tps, 
    float pos[3])
{
    int i, j;

    if (!tps) return;

    memcpy (new_pos, pos, 3 * sizeof(float));
    for (i = 0; i < tps->num_tps_nodes; i++) {
	float dist = sqrt (
	    ((pos[0] - tps[i].src[0]) * (pos[0] - tps[i].src[0])) +
	    ((pos[1] - tps[i].src[1]) * (pos[1] - tps[i].src[1])) +
	    ((pos[2] - tps[i].src[2]) * (pos[2] - tps[i].src[2])));
	dist = dist / tps[i].alpha;
	if (dist < 1.0) {
	    float weight = (1 - dist) * (1 - dist);
	    for (j = 0; j < 3; j++) {
		new_pos[j] += weight * tps[i].tgt[j];
	    }
	}
    }
}
#endif

void
tps_warp (
    Volume *vout,       /* Output image (sized and allocated) */
    Volume *vf_out,     /* Output vf (sized and allocated, can be null) */
    Tps_xform* tps, /* TPS control points */
    Volume *moving,     /* Input image */
    int linear_interp,  /* 1 = trilinear, 0 = nearest neighbors */
    float default_val   /* Fill in this value outside of image */
)
{
    int d;
    int vidx;
    float* vout_img = (float*) vout->img;

    int cpi;
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
	fprintf (stderr, "Error: tps_warp pix type mismatch\n");
	return;
    }
    if (vf_out && vf_out->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Error: tps_warp requires interleaved vf\n");
	return;
    }

    /* Set default */
    for (vidx = 0; vidx < vout->npix; vidx++) {
	vout_img[vidx] = default_val;
    }
    if (vf_out) {
	memset (vf_out->img, 0, vf_out->pix_size * vf_out->npix);
    }
	
    /* Loop through control points */
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
	    rmini = floorl (rmin - tps->img_origin[d] / tps->img_spacing[d]);
	    rmaxi = ceill (rmax - tps->img_origin[d] / tps->img_spacing[d]);

	    if (rmini < 0) rmini = 0;
	    if (rmaxi >= moving->dim[d]) rmaxi = moving->dim[d] - 1;

	    roi_offset[d] = rmini;
	    roi_size[d] = rmaxi - rmini + 1;
	}

	
    }

#if defined (commentout)
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
#endif
}

#if defined (commentout)
int
main (int argc, char *argv[])
{
    int i;
    FILE *src_fp, *tgt_fp, *tps_fp;

    /* Loop through points in source file */
    for (i = 0; i < 10; i++) {
	int rc;
	char src_buf[BUFLEN], tgt_buf[BUFLEN];
	int src_phase, tgt_phase;
	float src[3], tgt[3];
	if (!fgets (src_buf, BUFLEN, src_fp)) {
	    fprintf (stderr, "Ill-formed input file (1): %s\n", src_fn);
	    return 1;
	}
	if (!fgets (tgt_buf, BUFLEN, tgt_fp)) {
	    fprintf (stderr, "Ill-formed input file (1): %s\n", tgt_fn);
	    return 1;
	}
	rc = sscanf (src_buf, "%d %g %g %g", &src_phase, &src[0], 
		    &src[1],  &src[2]);
	if (rc != 4 || src_phase != i) {
	    fprintf (stderr, "Ill-formed input file (2): %s\n", src_fn);
	    return 1;
	}
	rc = sscanf (tgt_buf, "%d %g %g %g", &tgt_phase, &tgt[0], 
		    &tgt[1],  &tgt[2]);
	if (rc != 4 || tgt_phase != i) {
	    fprintf (stderr, "Ill-formed input file (2): %s\n", tgt_fn);
	    return 1;
	}

	if ((fabs (src[0] - tgt[0]) > MOTION_TOL) ||
	    (fabs (src[1] - tgt[1]) > MOTION_TOL) ||
	    (fabs (src[2] - tgt[2]) > MOTION_TOL)) {
	    float dist;
	    char tps_fn[256];

	    dist = sqrt (((src[0] - tgt[0]) * (src[0] - tgt[0]))
		    + ((src[1] - tgt[1]) * (src[1] - tgt[1]))
		    + ((src[2] - tgt[2]) * (src[2] - tgt[2])));

	    /* Append to fwd deformation */
	    snprintf (tps_fn, 256, "%s/%d_rbf_fwd.txt", deform_dir, i);
	    tps_fp = fopen (tps_fn, "ab");
	    if (!tps_fp) {
		fprintf (stderr, "Couldn't open file \"%s\" for read\n", tps_fn);
		return 1;
	    }
	    fprintf (tps_fp, "%g %g %g %g %g %g %g\n", 
		    src[0],
		    src[1],
		    src[2],
		    tgt[0]-src[0],
		    tgt[1]-src[1],
		    tgt[2]-src[2],
		    DIST_MULTIPLIER * dist
		    );
	    fclose (tps_fp);

	    /* Append to inv deformation */
	    snprintf (tps_fn, 256, "%s/%d_rbf_inv.txt", deform_dir, i);
	    tps_fp = fopen (tps_fn, "ab");
	    if (!tps_fp) {
		fprintf (stderr, "Couldn't open file \"%s\" for read\n", tps_fn);
		return 1;
	    }
	    fprintf (tps_fp, "%g %g %g %g %g %g %g\n", 
		    tgt[0],
		    tgt[1],
		    tgt[2],
		    src[0]-tgt[0],
		    src[1]-tgt[1],
		    src[2]-tgt[2],
		    DIST_MULTIPLIER * dist
		    );
	    fclose (tps_fp);
	}
    }

    fclose (src_fp);
    fclose (tgt_fp);
    return 0;
}
#endif
