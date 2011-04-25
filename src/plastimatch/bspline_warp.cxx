/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#if (SSE2_FOUND)
#include <xmmintrin.h>
#endif

#include "bspline.h"
#include "bspline_warp.h"
#include "interpolate.h"
#include "logfile.h"
#include "math_util.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "volume.h"
#include "volume_macros.h"

/* This only warps voxels within the ROI.  If you need the whole 
   image, call bspline_xform_extend. */
template <class T, int PIX_TYPE>
void
bspline_warp_internal (
    Volume *vout,       /* Output image (sized and allocated) */
    Volume *vf_out,     /* Output vf (sized and allocated, can be null) */
    Bspline_xform* bxf, /* Bspline transform coefficients */
    Volume *moving,     /* Input image */
    int linear_interp,  /* 1 = trilinear, 0 = nearest neighbors */
    T default_val       /* Fill in this value outside of image */
)
{
    int d;
    int vidx;
    T* vout_img = (T*) vout->img;

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
    T* m_img = (T*) moving->img;
    T m_val;

    /* A few sanity checks */
    if (vout->pix_type != PIX_TYPE) {
	print_and_exit ("Error: bspline_warp pix type mismatch\n");
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
	if (vout->spacing[d] != bxf->img_spacing[d]) {
	    print_and_exit ("Error: bspline_warp pix spacing mismatch\n");
	    return;
	}
    }
    if (vf_out && vf_out->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	print_and_exit ("Error: bspline_warp requires interleaved vf\n");
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

		/* If voxel is not inside moving image, continue. */
		if (!rc) continue;

		li_clamp_3d (mijk, mijk_f, mijk_r, li_1, li_2, moving);

		if (linear_interp) {
		    /* Find linear index of "corner voxel" in moving image */
		    mvf = INDEX_OF (mijk_f, moving->dim);

		    /* Macro is slightly faster than function */
		    /* Compute moving image intensity using linear 
		       interpolation */
		    LI_VALUE (m_val, 
			li_1[0], li_2[0],
			li_1[1], li_2[1],
			li_1[2], li_2[2],
			mvf, m_img, moving);

		    /* Assign warped value to output image */
		    vout_img[fv] = m_val;

		} else {
		    /* Find linear index of "nearest voxel" in moving image */
		    mvf = INDEX_OF (mijk_r, moving->dim);

		    /* Loop through planes */
		    /* Note: We omit looping through planes when linear 
		       interpolation is enabled, with the understanding 
		       that this is only used for warping structure sets */
		    for (int plane = 0; plane < moving->vox_planes; plane ++)
		    {
			/* Get moving image value */
			m_val = m_img[mvf*moving->vox_planes+plane];

			/* Assign to output image */
			vout_img[fv*moving->vox_planes+plane] = m_val;
		    }
		}
	    }
	}
    }
}

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
    switch (moving->pix_type)
    {
    case PT_UCHAR:
	bspline_warp_internal<unsigned char, PT_UCHAR> (
	    vout, vf_out, bxf, moving, linear_interp, default_val);
	break;
    case PT_SHORT:
	bspline_warp_internal<short, PT_SHORT> (
	    vout, vf_out, bxf, moving, linear_interp, default_val);
	break;
    case PT_UINT16:
	bspline_warp_internal<uint16_t, PT_UINT16> (
	    vout, vf_out, bxf, moving, linear_interp, default_val);
	break;
    case PT_UINT32:
	bspline_warp_internal<uint32_t, PT_UINT32> (
	    vout, vf_out, bxf, moving, linear_interp, default_val);
	break;
    case PT_FLOAT:
	bspline_warp_internal<float, PT_FLOAT> (
	    vout, vf_out, bxf, moving, linear_interp, default_val);
	break;
    case PT_VF_FLOAT_INTERLEAVED:
    case PT_VF_FLOAT_PLANAR:
	print_and_exit ("bspline_warp: sorry, this is not supported.\n");
	break;
    case PT_UCHAR_VEC_INTERLEAVED:
	bspline_warp_internal<unsigned char, PT_UCHAR_VEC_INTERLEAVED> (
	    vout, vf_out, bxf, moving, linear_interp, default_val);
	break;
    default:
	print_and_exit ("bspline_warp: sorry, this is not supported.\n");
	break;
    }
}
