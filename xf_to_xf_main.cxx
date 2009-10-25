/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"
#include "itkImage.h"
#include "itkLinearInterpolateImageFunction.h"

#include "getopt.h"
#include "xf_to_xf_main.h"
#include "plm_image_header.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "xform.h"

void
xf_to_xf_main (Xf_To_Xf_Parms* parms)
{
    Xform xf_in, xf_out;
    PlmImageHeader pih;

    load_xform (&xf_in, parms->xf_in_fn);
    pih.set_from_gpuit (parms->origin, parms->spacing, parms->dim, 0);

    switch (parms->xf_type) {
    case XFORM_NONE:
	print_and_exit ("Sorry, couldn't convert to XFORM_NONE\n");
	break;
    case XFORM_ITK_TRANSLATION:
	print_and_exit ("Sorry, couldn't convert to XFORM_ITK_TRANSLATION\n");
	break;
    case XFORM_ITK_VERSOR:
	print_and_exit ("Sorry, couldn't convert to XFORM_ITK_VERSOR\n");
	break;
    case XFORM_ITK_AFFINE:
	print_and_exit ("Sorry, couldn't convert to XFORM_ITK_AFFINE\n");
	break;
    case XFORM_ITK_BSPLINE:
	if (parms->grid_spac[0] <=0.0f) {
	    if (xf_in.m_type == XFORM_GPUIT_BSPLINE || xf_in.m_type == XFORM_ITK_BSPLINE) {
		/* Use grid spacing of input bspline */
		if (parms->nobulk) {
		    xform_to_itk_bsp_nobulk (&xf_out, &xf_in, &pih, 0);
		} else {
		    xform_to_itk_bsp (&xf_out, &xf_in, &pih, 0);
		}
	    } else {
		print_and_exit ("Sorry, grid spacing cannot be zero for conversion to itk_bsp\n");
	    }
	} else {
	    if (parms->nobulk) {
		xform_to_itk_bsp_nobulk (&xf_out, &xf_in, &pih, parms->grid_spac);
	    } else {
		xform_to_itk_bsp (&xf_out, &xf_in, &pih, parms->grid_spac);
	    }
	}
	break;
    case XFORM_ITK_TPS:
	print_and_exit ("Sorry, couldn't convert to XFORM_ITK_TPS\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	printf ("Converting to (itk) vector field\n");
	xform_to_itk_vf (&xf_out, &xf_in, &pih);
	break;
    case XFORM_GPUIT_BSPLINE:
#if defined (commentout)
	if (parms->grid_spac[0] <=0.0f) {
	    if (xf_in.m_type == XFORM_GPUIT_BSPLINE || xf_in.m_type == XFORM_ITK_BSPLINE) {
		xform_to_gpuit_bsp (&xf_out, &xf_in, &pih, 0);
	    } else {
		print_and_exit ("Sorry, grid spacing cannot be zero for conversion to gpuit_bsp\n");
	    }
	} else {
	    xform_to_gpuit_bsp (&xf_out, &xf_in, &pih, parms->grid_spac);
	}
#endif
	/* GPUIT_BSPLINE still requires separate bookkeeping for aux data. */
	print_and_exit ("Sorry, couldn't convert to XFORM_GPUIT_BSPLINE\n");
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	/* There would be no point of this, I think. */
	print_and_exit ("Sorry, couldn't convert to XFORM_GPUIT_VECTOR_FIELD\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
    save_xform (&xf_out, parms->xf_out_fn);
}

void
print_usage (void)
{
    printf ("Usage: xf_to_xf --output-type=type --input=xform_in --output=vf_out --dims=\"x y z\"\n");
    printf ("          --origin=\"x y z\" --spacing=\"x y z\" --grid-spacing=\"x y z\"\n");
    printf ("       Supported output-types: vf, itk_bsp.\n");
    exit (-1);
}

void
parse_args (Xf_To_Xf_Parms* parms, int argc, char* argv[])
{
    int ch, rc;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           1 },
	{ "output",         required_argument,      NULL,           2 },
	{ "output-type",    required_argument,      NULL,           3 },
	{ "dims",           required_argument,      NULL,           4 },
	{ "origin",         required_argument,      NULL,           5 },
	{ "spacing",        required_argument,      NULL,           6 },
	{ "grid-spacing",   required_argument,      NULL,           7 },
	{ "nobulk",         no_argument,            NULL,           8 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->xf_in_fn, optarg, _MAX_PATH);
	    break;
	case 2:
	    strncpy (parms->xf_out_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    if (!strcmp (optarg, "vf")) {
		parms->xf_type = XFORM_ITK_VECTOR_FIELD;
	    } else if (!strcmp (optarg, "itk_bsp")) {
		printf ("CVT to itk_bsp\n");
		parms->xf_type = XFORM_ITK_BSPLINE;
	    } else if (!strcmp (optarg, "gpuit_bsp")) {
		printf ("CVT to gpuit_bsp\n");
		parms->xf_type = XFORM_GPUIT_BSPLINE;
	    } else {
		fprintf (stderr, "Unexpected output type.  Hmm, what to do...\nAborting.\n");
		print_usage();
	    }
	    break;
	case 4: {
		rc = sscanf (optarg, "%d %d %d", &(parms->dim[0]), 
			&(parms->dim[1]), &(parms->dim[2]));
		if (rc != 3) {
		    print_usage();
		}
	    }
	    break;
	case 5:
	    rc = sscanf (optarg, "%g %g %g", &(parms->origin[0]), 
		    &(parms->origin[1]), &(parms->origin[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 6:
	    rc = sscanf (optarg, "%g %g %g", &(parms->spacing[0]), 
		    &(parms->spacing[1]), &(parms->spacing[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 7:
	    rc = sscanf (optarg, "%g %g %g", &(parms->grid_spac[0]), 
		    &(parms->grid_spac[1]), &(parms->grid_spac[2]));
	    if (rc != 3) {
		print_usage();
	    }
	    break;
	case 8:
	    parms->nobulk = 1;
	    break;
	default:
	    break;
	}
    }
    if (!parms->xf_in_fn[0] || !parms->xf_out_fn[0] || !parms->xf_type || !parms->dim[0] || parms->spacing[0] == 0.0) {
	printf ("Error: must specify all options\n");
	print_usage();
    }
}

int
main(int argc, char *argv[])
{
    Xf_To_Xf_Parms parms;
    
    parse_args (&parms, argc, argv);

    xf_to_xf_main (&parms);

    printf ("Finished!\n");
    return 0;
}
