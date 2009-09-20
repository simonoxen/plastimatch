/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"
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

	if (parms->fixed_im_fn[0]) {
	    /* if given, use the grid spacing of user-supplied fixed image */
	    FloatImageType::Pointer fixed = load_float (parms->fixed_im_fn, 0);
	    pih.set_from_itk_image (fixed);
	} else {
	    /* otherwise, use the grid spacing of the input image */
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

void
print_usage (void)
{
    printf ("Usage: plastimatch warp [options]\n"
	    "Required:\n"
	    "    --input=filename\n"
	    "    --xf=filename\n"
	    "    --output=filename\n"
	    "Optional:\n"
	    "    --interpolation=nn\n"
	    "    --fixed=filename\n"
	    "    --output_vf=filename\n"
	    "    --default_val=number\n"
	    "    --output-format=dicom\n");
    exit (-1);
}

void
parse_args (Warp_Parms* parms, int argc, char* argv[])
{
    int ch;
    int rc;
    int have_offset = 0;
    int have_spacing = 0;
    int have_dims = 0;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "vf",             required_argument,      NULL,           4 },
	{ "default_val",    required_argument,      NULL,           5 },
	{ "xf",             required_argument,      NULL,           6 },
	{ "fixed",	    required_argument,      NULL,           7 },
	{ "output_vf",      required_argument,      NULL,           8 },
	{ "interpolation",  required_argument,      NULL,           9 },
	{ "offset",         required_argument,      NULL,           10 },
	{ "spacing",        required_argument,      NULL,           11 },
	{ "dims",           required_argument,      NULL,           12 },
	{ "output-format",  required_argument,      NULL,           13 },
	{ NULL,             0,                      NULL,           0 }
    };

    /* Skip command */
    optind ++;

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->mha_out_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (parms->vf_in_fn, optarg, _MAX_PATH);
	    break;
	case 5:
	    if (sscanf (optarg, "%f", &parms->default_val) != 1) {
		printf ("Error: default_val takes an argument\n");
		print_usage();
	    }
	    break;
	case 6:
	    strncpy (parms->xf_in_fn, optarg, _MAX_PATH);
	    break;
	case 7:
	    strncpy (parms->fixed_im_fn, optarg, _MAX_PATH);
	    break;
	case 8:
	    strncpy (parms->vf_out_fn, optarg, _MAX_PATH);
	    break;
	case 9:
	    if (!strcmp (optarg, "nn")) {
		parms->interp_lin = 0;
	    } else if (!strcmp (optarg, "linear")) {
		parms->interp_lin = 1;
	    } else {
		fprintf (stderr, "Error.  --interpolation must be either nn or linear.\n");
		print_usage ();
	    }
	    break;
	case 10:
	    rc = sscanf (optarg, "%f %f %f", &parms->offset[0], &parms->offset[1], &parms->offset[2]);
	    if (rc != 3) {
		fprintf (stderr, "Error.  --offset requires 3 values.");
		print_usage ();
	    }
	    have_offset = 1;
	    break;
	case 11:
	    rc = sscanf (optarg, "%f %f %f", &parms->spacing[0], &parms->spacing[1], &parms->spacing[2]);
	    if (rc != 3) {
		fprintf (stderr, "Error.  --spacing requires 3 values.");
		print_usage ();
	    }
	    have_spacing = 1;
	    break;
	case 12:
	    rc = sscanf (optarg, "%d %d %d", &parms->dims[0], &parms->dims[1], &parms->dims[2]);
	    if (rc != 3) {
		fprintf (stderr, "Error.  --dims requires 3 values.");
		print_usage ();
	    }
	    have_dims = 1;
	    break;
	case 13:
	    if (!strcmp (optarg, "dicom")) {
		parms->output_dicom = 1;
	    } else {
		fprintf (stderr, "Error.  --output-type option only supports dicom.\n");
		print_usage ();
	    }
	    break;
	default:
	    break;
	}
    }
    if (!parms->mha_in_fn[0] || !parms->mha_out_fn[0] || !(parms->vf_in_fn[0] || parms->xf_in_fn[0])) {
	print_usage();
    }
}

#if defined (commentout)
void
do_command_warp (int argc, char* argv[])
//int
//main (int argc, char *argv[])
{
    Warp_Parms parms;
    
    parse_args (&parms, argc, argv);

    warp_image_main (&parms);

    printf ("Finished!\n");
//    return 0;
}
#endif
