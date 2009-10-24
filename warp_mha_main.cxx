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
static void
do_warp_itk (
    T *im_warped,                         /* Output　*/
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    T im_in,                              /* Input */
    U output_type                         /* Input */
)
{
    Xform xform_tmp;
    printf ("converting to vf...\n");
    xform_to_itk_vf (&xform_tmp, xf_in, pih);
    *vf = xform_tmp.get_itk_vf ();

    printf ("Warping...\n");
    *im_warped = itk_warp_image (im_in, *vf, parms->interp_lin, 
				 (U) parms->default_val);
}

/* Fallback for unsupported native warping */
template<class T, class U>
static void
do_warp_native (
    T *im_warped,                         /* Output　*/
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    T im_in,                              /* Input */
    U output_type                         /* Input */
)
{
    printf ("Tried to warp native, but falling back to itk\n");
    do_warp_itk (im_warped, vf, parms, xf_in, pih, im_in, output_type);
}

#if defined (commentout)
/* Native warping for floats */
template<class T>
static void
do_warp_native (
    T *im_warped,                         /* Output　*/
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    T im_in,                              /* Input */
    float output_type                     /* Input */
)
{
    printf ("Hello world\n");
    exit (-1);
}
#endif

template<class T, class U>
static void
do_warp (
    T *im_warped,                         /* Output　*/
    DeformationFieldType::Pointer *vf,    /* Output */
    Warp_Parms* parms,                    /* Input */
    Xform *xf_in,                         /* Input */
    PlmImageHeader *pih,                  /* Input */
    T im_in,                              /* Input */
    U output_type                         /* Input */
)
{
    /* If user wants ITK-based warping, respect their wish */
    if (parms->use_itk) {
	do_warp_itk (im_warped, vf, parms, xf_in, pih, im_in, output_type);
	return;
    }

    /* Otherwise, we try to do native warping (when possible) */
    switch (xf_in->m_type) {
    case XFORM_GPUIT_BSPLINE:
	do_warp_native (im_warped, vf, parms, xf_in, pih, im_in, output_type);
	break;
    default:
	do_warp_itk (im_warped, vf, parms, xf_in, pih, im_in, output_type);
	break;
    }
}

/* GCS FIX: This function can't actually change the output type */
template<class T, class U>
static void 
warp_any (Warp_Parms* parms, T im_in, U output_type)
{
    PlmImageHeader pih;
    Xform xform;

    T im_warped = T::ObjectType::New();
    DeformationFieldType::Pointer vf = DeformationFieldType::New();

    printf ("Loading xform (%s)\n", parms->xf_in_fn);
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

    /* Do the warp */
    do_warp (&im_warped, &vf, parms, &xform, &pih, im_in, output_type);

    /* Save output files */
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

