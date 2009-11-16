/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>

#include "plm_int.h"
#include "getopt.h"
#include "warp_main.h"
#include "plm_image_header.h"
#include "itk_image.h"
#include "itk_warp.h"
#include "print_and_exit.h"
#include "xform.h"
#include "readmha.h"
#include "volume.h"

static void
do_warp_itk (
    PlmImage *im_warped,                  /* Output */
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    PlmImage *im_in                       /* Input */
)
{
    Xform xform_tmp;
    printf ("converting to vf...\n");
    xform_to_itk_vf (&xform_tmp, xf_in, pih);
    *vf = xform_tmp.get_itk_vf ();

    printf ("Warping...\n");
    switch (im_in->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	im_warped->m_itk_uchar = itk_warp_image (
	    im_in->m_itk_uchar, 
	    *vf, 
	    parms->interp_lin, 
	    static_cast<unsigned char>(parms->default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_UCHAR;
	im_warped->m_type = PLM_IMG_TYPE_ITK_UCHAR;
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	im_warped->m_itk_short = itk_warp_image (
	    im_in->m_itk_short, 
	    *vf, 
	    parms->interp_lin, 
	    static_cast<short>(parms->default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_SHORT;
	im_warped->m_type = PLM_IMG_TYPE_ITK_SHORT;
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	im_warped->m_itk_ushort = itk_warp_image (
	    im_in->m_itk_ushort, 
	    *vf, 
	    parms->interp_lin, 
	    static_cast<unsigned short>(parms->default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_USHORT;
	im_warped->m_type = PLM_IMG_TYPE_ITK_USHORT;
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	im_warped->m_itk_uint32 = itk_warp_image (
	    im_in->m_itk_uint32, 
	    *vf, 
	    parms->interp_lin, 
	    static_cast<uint32_t>(parms->default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_ULONG;
	im_warped->m_type = PLM_IMG_TYPE_ITK_ULONG;
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	im_warped->m_itk_float = itk_warp_image (
	    im_in->m_itk_float, 
	    *vf, 
	    parms->interp_lin, 
	    static_cast<float>(parms->default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_FLOAT;
	im_warped->m_type = PLM_IMG_TYPE_ITK_FLOAT;
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
	im_warped->m_itk_double = itk_warp_image (
	    im_in->m_itk_double, 
	    *vf, 
	    parms->interp_lin, 
	    static_cast<double>(parms->default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_DOUBLE;
	im_warped->m_type = PLM_IMG_TYPE_ITK_DOUBLE;
	break;
    case PLM_IMG_TYPE_ITK_CHAR:
    case PLM_IMG_TYPE_ITK_LONG:
    default:
	print_and_exit ("Unhandled case in do_warp_itk()\n");
	break;
    }
}

/* Native warping (only gpuit bspline + float) */
static void
do_warp_native (
    PlmImage *im_warped,                  /* Output */
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    PlmImage *im_in                       /* Input */
)
{
    Xform xf_tmp;
    Xform vf_tmp;
    BSPLINE_Xform* bxf_in = xf_in->get_gpuit_bsp ();
    Volume *vf_out = 0;     /* Output vector field */
    Volume *v_out = 0;       /* Output warped image */
    int dim[3];
    float origin[3];
    float spacing[3];
    float direction_cosines[9];

    /* Convert input image to gpuit format */
    printf ("Converting input image...\n");
    Volume *v_in = im_in->gpuit_float();

    /* Transform input xform to gpuit bspline with correct voxel spacing */
    printf ("Converting xform...\n");
    xform_to_gpuit_bsp (&xf_tmp, xf_in, pih, bxf_in->grid_spac);

    /* Create output vf */
    pih->get_gpuit_origin (origin);
    pih->get_gpuit_spacing (spacing);
    pih->get_gpuit_dim (dim);
    pih->get_gpuit_direction_cosines (direction_cosines);
    if (parms->vf_out_fn[0]) {
	printf ("Creating output vf...\n");
	vf_out = volume_create (dim, origin, spacing, PT_VF_FLOAT_INTERLEAVED,
				direction_cosines, 0);
    }

    /* Create output image */
    printf ("Creating output volume...\n");
    v_out = volume_create (dim, origin, spacing, PT_FLOAT, 
			   direction_cosines, 0);

    /* Warp using gpuit native warper */
    bspline_warp (v_out, vf_out, xf_tmp.get_gpuit_bsp(), v_in, 
		  parms->interp_lin, 
		  parms->default_val);

    /* Return output image to caller */
    im_warped->set_gpuit_float (v_out);

    /* Bspline_warp only operates on float.  We need to back-convert */
    im_warped->convert_to_original_type ();

    /* Return vf to caller */
    if (parms->vf_out_fn[0]) {
	*vf = xform_gpuit_vf_to_itk_vf (vf_out, 0);
	volume_free (vf_out);
    }
}

static void
do_warp (
    PlmImage *im_warped,                  /* Output */
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    PlmImage *im_in                       /* Input */
)
{
    /* If user requested ITK-based warping, respect their wish */
    if (parms->use_itk) {
	do_warp_itk (im_warped, vf, parms, xf_in, pih, im_in);
	return;
    }

    /* Otherwise, try to do native warping where possible */
    if (xf_in->m_type == XFORM_GPUIT_BSPLINE) {
	switch (im_in->m_type) {
	case PLM_IMG_TYPE_ITK_SHORT:
	case PLM_IMG_TYPE_ITK_FLOAT:
	case PLM_IMG_TYPE_ITK_ULONG:
	    do_warp_native (im_warped, vf, parms, xf_in, pih, im_in);
	    break;
	default:
	    do_warp_itk (im_warped, vf, parms, xf_in, pih, im_in);
	    break;
	}
    } else {
	do_warp_itk (im_warped, vf, parms, xf_in, pih, im_in);
    }
}

void
warp_image_main (Warp_Parms* parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    PlmImage im_in, im_out;
    PlmImage* im_out_ptr;
    PlmImageHeader pih;
    Xform xform;

    /* Load input image */
    im_in.load_native (parms->mha_in_fn);

    /* Load transform */
    if (parms->xf_in_fn[0]) {
	printf ("Loading xform (%s)\n", parms->xf_in_fn);
	load_xform (&xform, parms->xf_in_fn);
    }

    /* Try to guess the proper dimensions and spacing for output image */
    if (parms->fixed_im_fn[0]) {
	/* use the spacing of user-supplied fixed image */
	FloatImageType::Pointer fixed = load_float (parms->fixed_im_fn, 0);
	pih.set_from_itk_image (fixed);
    } else if (xform.m_type == XFORM_ITK_VECTOR_FIELD) {
	/* use the spacing from input vector field */
	pih.set_from_itk_image (xform.get_itk_vf());
    } else {
	/* otherwise, use the spacing of the input image */
	pih.set_from_plm_image (&im_in);
    }

    printf ("PIH is:\n");
    pih.print ();

    /* Do the warp */
    if (parms->xf_in_fn[0]) {
	do_warp (&im_out, &vf, parms, &xform, &pih, &im_in);
	im_out_ptr = &im_out;
    } else {
	im_out_ptr = &im_in;
    }

    /* Save output image */
    printf ("Saving image...\n");
    if (parms->output_dicom) {
	im_out_ptr->save_short_dicom (parms->mha_out_fn);
    } else {
	im_out_ptr->save_image (parms->mha_out_fn);
    }

    /* Save output vector field */
    if (parms->xf_in_fn[0] && parms->vf_out_fn[0]) {
	printf ("Saving vf...\n");
	itk_image_save (vf, parms->vf_out_fn);
    }
}
