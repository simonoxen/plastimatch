/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*  Warp the moving image into the space of the fixed image based 
    on the vector field. It takes 3 inputs: floating image, vector
    field, and an output image file names */
#include <time.h>
#include "config.h"
#include "itkImage.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkCastImageFilter.h"

#include "getopt.h"
#include "warp_mha_main.h"
#include "itk_image.h"
#include "itk_warp.h"
#include "print_and_exit.h"
#include "xform.h"
#include "readmha.h"
#include "volume.h"

void
warp_image_main (Warp_Parms* parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
	
    if (parms->output_type == TYPE_SHORT) {
#if defined (commentout)
	{
	    Volume *vin, *vf, *vout;
	    vin = read_mha (parms->mha_in_fn);
	    volume_convert_to_float (vin);
	    vf = read_mha (parms->vf_in_fn);
	    vout = volume_warp (0, vin, vf);
	    write_mha (parms->mha_out_fn, vout);
	    return;
	}
#endif
	ShortImageType::Pointer im = ShortImageType::New();
	ShortImageType::Pointer im_warped = ShortImageType::New();
	printf ("Loading...\n");
	im = load_short (parms->mha_in_fn);

	if (parms->vf_in_fn[0]) {
	    printf ("Loading vf...\n");
	    vf = load_float_field (parms->vf_in_fn);
			
	    printf ("Warping...\n");
	    im_warped = itk_warp_image (im, im, vf, parms->interp_lin, parms->default_val);
	} else { /* need to convert the deformation parameters into vector fields */
	    printf ("Loading deformation parameters...\n");
	    Xform xform, xform_tmp;
	    load_xform (&xform, parms->xf_in_fn);
		
	    printf ("converting to vf...\n");

	    typedef itk::CastImageFilter <ShortImageType, FloatImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();

	    if (parms->fixed_im_fn[0]) { /* if given, use the grid spacing of the fixed image */
		ShortImageType::Pointer fixed = ShortImageType::New();
		fixed = load_short (parms->fixed_im_fn);
		caster->SetInput(fixed);
		caster->Update();
		xform_to_itk_vf(&xform_tmp, &xform, caster->GetOutput());
		vf = xform_tmp.get_itk_vf();

		printf ("Warping...\n");
		im_warped = itk_warp_image (im, fixed, vf, parms->interp_lin, parms->default_val);
	    } else {
		caster->SetInput(im);
		caster->Update();
		xform_to_itk_vf(&xform_tmp, &xform, caster->GetOutput());
		vf = xform_tmp.get_itk_vf();

		printf ("Warping...\n");
		im_warped = itk_warp_image (im, im, vf, parms->interp_lin, parms->default_val);
	    }
	}

	printf ("Saving...\n");
	save_image (im_warped, parms->mha_out_fn);
	if (parms->vf_out_fn[0])
	    save_image(vf, parms->vf_out_fn);
    } else if (parms->output_type == TYPE_FLOAT) {
	FloatImageType::Pointer im = FloatImageType::New();
	FloatImageType::Pointer im_warped = FloatImageType::New();
	printf ("Loading...\n");
	im = load_float (parms->mha_in_fn);
		
	if (parms->vf_in_fn[0]) {
	    printf ("Loading vf...\n");
	    vf = load_float_field (parms->vf_in_fn);
	    printf ("Warping...\n");
	    im_warped = itk_warp_image (im, im, vf, parms->interp_lin, parms->default_val);
	} else { /* need to convert the deformation parameters into vector fields */
	    printf ("Loading deformation parameters...\n");
	    Xform xform, xform_tmp;
	    load_xform (&xform, parms->xf_in_fn);
		
	    printf ("converting to vf...\n");

	    if (parms->fixed_im_fn[0]) { /* if given, use the grid spacing of the fixed image */
		FloatImageType::Pointer fixed = FloatImageType::New();
		fixed = load_float (parms->fixed_im_fn);
		xform_to_itk_vf(&xform_tmp, &xform, fixed);
		vf = xform_tmp.get_itk_vf();
		printf ("Warping...\n");
		im_warped = itk_warp_image (im, fixed, vf, parms->interp_lin, parms->default_val);
	    } else {
		xform_to_itk_vf(&xform_tmp, &xform, im);
		vf = xform_tmp.get_itk_vf();
		printf ("Warping...\n");
		im_warped = itk_warp_image (im, im, vf, parms->interp_lin, parms->default_val);
	    }
	}

	printf ("Saving...\n");
	save_image (im_warped, parms->mha_out_fn);
	if (parms->vf_out_fn[0]) 
	    save_image(vf, parms->vf_out_fn);
    } else {
	printf ("Error, unsupported output type\n");
	exit (-1);
    }
}

void
print_usage (void)
{
    printf ("Usage: warp_mha --input=image_in --vf=vf_in --output=image_out --output_type=type [--interpolation nn]\n");
    printf ("   or: warp_mha --input=image_in --deform_parm=xf_in (--fixed=fixed_im_fn) --output=image_out --output_type=type [--interpolation nn]\n");
    exit (-1);
}

void
parse_args (Warp_Parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "output_type",    required_argument,      NULL,           1 },
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "vf",             required_argument,      NULL,           4 },
	{ "default_val",    required_argument,      NULL,           5 },
	{ "deform_parm",    required_argument,      NULL,           6 },
	{ "fixed",	    required_argument,      NULL,           7 },
	{ "output_vf",      required_argument,      NULL,           8 },
	{ "interpolation",  required_argument,      NULL,           9 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    if (!strcmp(optarg,"ushort") || !strcmp(optarg,"unsigned")) {
		parms->output_type = TYPE_USHORT;
	    }
	    else if (!strcmp(optarg,"short") || !strcmp(optarg,"signed")) {
		parms->output_type = TYPE_SHORT;
	    }
	    else if (!strcmp(optarg,"float")) {
		parms->output_type = TYPE_FLOAT;
	    }
	    else if (!strcmp(optarg,"mask") || !strcmp(optarg,"uchar")) {
		parms->output_type = TYPE_UCHAR;
	    }
	    else if (!strcmp(optarg,"vf")) {
		parms->output_type = TYPE_FLOAT_FIELD;
	    }
	    else {
		print_usage();
	    }
	    break;
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
	default:
	    break;
	}
    }
    if (!parms->mha_in_fn[0] || !parms->mha_out_fn[0] || !(parms->vf_in_fn[0] || parms->xf_in_fn[0])) {
		printf ("Error: must specify --input and --output and --vf or --deform_parm\n");
		print_usage();
    }
    if (parms->output_type == TYPE_UNSPECIFIED) {
		printf ("Error: must specify --output_type\n");
		print_usage();
    }
}


int
main(int argc, char *argv[])
{
    Warp_Parms parms;
    
    parse_args (&parms, argc, argv);

    warp_image_main (&parms);

    printf ("Finished!\n");
    return 0;
}
