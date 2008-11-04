/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "getopt.h"
#include "warp_pointset_main.h"
#if defined (commentout)
#include "itkImage.h"
#include "itkWarpImageFilter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkCastImageFilter.h"

#include "warp_mha_main.h"
#include "itk_image.h"
#include "itk_warp.h"
#include "print_and_exit.h"
#include "xform.h"
#include "readmha.h"
#include "volume.h"
#endif

#if defined (commentout)
template<class T, class U>
void warp_any (Warp_Parms* parms, T im_in, U)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    T im_warped = T::ObjectType::New();
    T im_ref = im_in;

    if (parms->vf_in_fn[0]) {
	printf ("Loading vf...\n");
	vf = load_float_field (parms->vf_in_fn);
    		
	printf ("Warping...\n");
	im_warped = itk_warp_image (im_in, vf, parms->interp_lin, (U) parms->default_val);

    } else {
	/* convert xform into vector field, then warp */
	PlmImageHeader pih;

	printf ("Loading xform...\n");
	Xform xform, xform_tmp;
	load_xform (&xform, parms->xf_in_fn);

	if (parms->fixed_im_fn[0]) {
	    /* if given, use the grid spacing of user-supplied fixed image */
	    FloatImageType::Pointer fixed = load_float (parms->fixed_im_fn);
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
    }

    printf ("Saving...\n");
    save_image (im_warped, parms->mha_out_fn);
    if (parms->vf_out_fn[0]) {
	save_image(vf, parms->vf_out_fn);
    }
}
#endif

void
warp_pointset_main (Warp_Pointset_Parms* parms)
{
}

void
print_usage (void)
{
    printf ("Usage: warp_pointset --input=ps_in  --xf=xf_in --output=ps_out\n");
    exit (-1);
}

void
parse_args (Warp_Pointset_Parms* parms, int argc, char* argv[])
{
    int ch;
    int rc;
    int have_offset = 0;
    int have_spacing = 0;
    int have_dims = 0;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "xf",             required_argument,      NULL,           4 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->ps_in_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->ps_out_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (parms->xf_in_fn, optarg, _MAX_PATH);
	    break;
	default:
	    break;
	}
    }
    if (!parms->ps_in_fn[0] || !parms->ps_out_fn[0] || !parms->xf_in_fn[0]) {
	printf ("Error: must specify --input, --output, and --vf or --xf\n");
	print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Warp_Pointset_Parms parms;
    
    parse_args (&parms, argc, argv);

    warp_pointset_main (&parms);

    printf ("Finished!\n");
    return 0;
}
