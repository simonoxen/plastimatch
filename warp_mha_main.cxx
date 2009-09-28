/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"
#include "plm_int.h"
#include "itkImage.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkCastImageFilter.h"

#include "getopt.h"
#include "warp_main.h"
#include "itk_image.h"
#include "itk_warp.h"
#include "print_and_exit.h"
#include "xform.h"
#include "readmha.h"
#include "volume.h"

template<class T, class U>
void warp_any (Warp_Parms* parms, T im_in, U)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    T im_warped = T::ObjectType::New();
    T im_ref = im_in;

#if defined (commentout)
    if (parms->vf_in_fn[0]) {
	printf ("Loading vf...\n");
	vf = load_float_field (parms->vf_in_fn);
    		
	printf ("Warping...\n");
	im_warped = itk_warp_image (im_in, vf, parms->interp_lin, (U) parms->default_val);

    } else {
#endif
	/* convert xform into vector field, then warp */
	PlmImageHeader pih;

	printf ("Loading xform...\n");
	Xform xform, xform_tmp;
	load_xform (&xform, parms->xf_in_fn);

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
	    pih.set_from_itk_image (im_in);
	}
	printf ("converting to vf...\n");
 	xform_to_itk_vf (&xform_tmp, &xform, &pih);
	vf = xform_tmp.get_itk_vf();

	printf ("Warping...\n");
	im_warped = itk_warp_image (im_in, vf, parms->interp_lin, (U) parms->default_val);
#if defined (commentout)
    }
#endif

    printf ("Saving...\n");
    if (parms->output_dicom) {
	save_short_dicom (im_warped, parms->mha_out_fn);
    } else {
	save_image (im_warped, parms->mha_out_fn);
    }
    if (parms->vf_out_fn[0]) {
	save_image(vf, parms->vf_out_fn);
    }
}

void
warp_image_main (Warp_Parms* parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();

    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;

    itk__GetImageType (parms->mha_in_fn, pixelType, componentType);

    switch (componentType) {
    case itk::ImageIOBase::UCHAR:
	{
	    UCharImageType::Pointer mha_in 
		    = load_uchar (parms->mha_in_fn, 0);
	    warp_any (parms, mha_in, static_cast<unsigned char>(0));
	}
	break;
    case itk::ImageIOBase::SHORT:
	{
	    ShortImageType::Pointer mha_in 
		    = load_short (parms->mha_in_fn, 0);
	    warp_any (parms, mha_in, static_cast<short>(0));
	}
	break;
#if (CMAKE_SIZEOF_UINT == 4)
    case itk::ImageIOBase::UINT:
#endif
#if (CMAKE_SIZEOF_ULONG == 4)
    case itk::ImageIOBase::ULONG:
#endif
	{
	    UInt32ImageType::Pointer mha_in 
		    = load_uint32 (parms->mha_in_fn, 0);
	    warp_any (parms, mha_in, static_cast<uint32_t>(0));
	}
	break;
    case itk::ImageIOBase::FLOAT:
	{
	    FloatImageType::Pointer mha_in 
		    = load_float (parms->mha_in_fn, 0);
	    warp_any (parms, mha_in, static_cast<float>(0));
	}
	break;
    default:
	printf ("Error, unsupported output type\n");
	exit (-1);
	break;
    }
}

