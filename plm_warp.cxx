/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>

#include "itk_image.h"
#include "itk_warp.h"
#include "mha_io.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_warp.h"
#include "print_and_exit.h"
#include "volume.h"
#include "xform.h"

static void
plm_warp_itk (
    Plm_image *im_warped,                    /* Output (optional) */
    DeformationFieldType::Pointer *vf_out,   /* Output (optional) */
    Xform *xf_in,                            /* Input */
    Plm_image_header *pih,                   /* Input */
    Plm_image *im_in,                        /* Input */
    float default_val,     /* Input:  Value for pixels without match */
    int interp_lin         /* Input:  Trilinear (1) or nn (0) */
)
{
    Xform xform_tmp;
    DeformationFieldType::Pointer vf;

    /* Create an itk vector field from xf_in */
    xform_to_itk_vf (&xform_tmp, xf_in, pih);
    vf = xform_tmp.get_itk_vf ();

    /* If caller wants the vf, we assign it here */
    if (vf_out) {
	*vf_out = vf;
    }

    /* If caller only wants the vf, we are done */
    if (!im_warped) {
	return;
    }

    /* Warp the image */
    printf ("Warping...\n");
    switch (im_in->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	im_warped->m_itk_uchar = itk_warp_image (
	    im_in->m_itk_uchar, 
	    vf, 
	    interp_lin, 
	    static_cast<unsigned char>(default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_UCHAR;
	im_warped->m_type = PLM_IMG_TYPE_ITK_UCHAR;
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	im_warped->m_itk_short = itk_warp_image (
	    im_in->m_itk_short, 
	    vf, 
	    interp_lin, 
	    static_cast<short>(default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_SHORT;
	im_warped->m_type = PLM_IMG_TYPE_ITK_SHORT;
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	im_warped->m_itk_ushort = itk_warp_image (
	    im_in->m_itk_ushort, 
	    vf, 
	    interp_lin, 
	    static_cast<unsigned short>(default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_USHORT;
	im_warped->m_type = PLM_IMG_TYPE_ITK_USHORT;
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	im_warped->m_itk_uint32 = itk_warp_image (
	    im_in->m_itk_uint32, 
	    vf, 
	    interp_lin, 
	    static_cast<uint32_t>(default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_ULONG;
	im_warped->m_type = PLM_IMG_TYPE_ITK_ULONG;
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	im_warped->m_itk_float = itk_warp_image (
	    im_in->m_itk_float, 
	    vf, 
	    interp_lin, 
	    static_cast<float>(default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_FLOAT;
	im_warped->m_type = PLM_IMG_TYPE_ITK_FLOAT;
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
	im_warped->m_itk_double = itk_warp_image (
	    im_in->m_itk_double, 
	    vf, 
	    interp_lin, 
	    static_cast<double>(default_val));
	im_warped->m_original_type = PLM_IMG_TYPE_ITK_DOUBLE;
	im_warped->m_type = PLM_IMG_TYPE_ITK_DOUBLE;
	break;
    case PLM_IMG_TYPE_ITK_CHAR:
    case PLM_IMG_TYPE_ITK_LONG:
    default:
	print_and_exit ("Unhandled case in plm_warp_itk()\n");
	break;
    }
}

/* Native warping (only gpuit bspline + float) */
static void
plm_warp_native (
    Plm_image *im_warped,                 /* Output */
    DeformationFieldType::Pointer *vf,    /* Output */
    Xform *xf_in,                         /* Input */
    Plm_image_header *pih,                /* Input */
    Plm_image *im_in,                     /* Input */
    float default_val,     /* Input:  Value for pixels without match */
    int interp_lin         /* Input:  Trilinear (1) or nn (0) */
)
{
    Xform xf_tmp;
    Xform vf_tmp;
    Bspline_xform* bxf_in = xf_in->get_gpuit_bsp ();
    Volume *vf_out = 0;     /* Output vector field */
    Volume *v_out = 0;      /* Output warped image */
    int dim[3];
    float origin[3];
    float spacing[3];
    float direction_cosines[9];

    /* Convert input image to gpuit format */
    printf ("Running: plm_warp_native\n");
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
    if (vf) {
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
	interp_lin, default_val);

    /* Return output image to caller */
    im_warped->set_gpuit (v_out);

    /* Bspline_warp only operates on float.  We need to back-convert */
    im_warped->convert (im_in->m_original_type);
    im_warped->m_original_type = im_in->m_original_type;

    /* Return vf to caller */
    if (vf) {
	*vf = xform_gpuit_vf_to_itk_vf (vf_out, 0);
	volume_destroy (vf_out);
    }
}

void
plm_warp (
    Plm_image *im_warped,  /* Output: Output image */
    DeformationFieldType::Pointer* vf,    /* Output: Output vf (optional) */
    Xform *xf_in,          /* Input:  Input image warped by this xform */
    Plm_image_header *pih, /* Input:  Size of output image */
    Plm_image *im_in,      /* Input:  Input image */
    float default_val,     /* Input:  Value for pixels without match */
    int use_itk,           /* Input:  Force use of itk (1) or not (0) */
    int interp_lin         /* Input:  Trilinear (1) or nn (0) */
)
{
    /* If user requested ITK-based warping, respect their wish */
    if (use_itk) {
	plm_warp_itk (im_warped, vf, xf_in, pih, im_in, default_val,
	    interp_lin);
	return;
    }

    /* Otherwise, try to do native warping where possible */
    if (xf_in->m_type == XFORM_GPUIT_BSPLINE) {
	switch (im_in->m_type) {
	case PLM_IMG_TYPE_ITK_UCHAR:
	case PLM_IMG_TYPE_ITK_SHORT:
	case PLM_IMG_TYPE_ITK_ULONG:
	case PLM_IMG_TYPE_ITK_FLOAT:
	case PLM_IMG_TYPE_GPUIT_UCHAR:
	case PLM_IMG_TYPE_GPUIT_SHORT:
	case PLM_IMG_TYPE_GPUIT_UINT32:
	case PLM_IMG_TYPE_GPUIT_FLOAT:
	    if (im_in->m_type == PLM_IMG_TYPE_GPUIT_SHORT) {
		printf ("Image type = GPUIT_SHORT\n");
	    } else if (im_in->m_type == PLM_IMG_TYPE_GPUIT_UINT32) {
		printf ("Image type = GPUIT_UINT32\n");
	    }
	    plm_warp_native (im_warped, vf, xf_in, pih, im_in, default_val,
		interp_lin);
	    break;
	default:
	    plm_warp_itk (im_warped, vf, xf_in, pih, im_in, default_val,
		interp_lin);
	    break;
	}
    } else {
	plm_warp_itk (im_warped, vf, xf_in, pih, im_in, default_val,
	    interp_lin);
    }
}
