/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plm_config.h"

#include "plm_image_header.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "xform.h"
#include "xform_convert.h"

void 
xform_convert (Xform_convert *xfc)
{
    Plm_image_header pih;
    pih.set_from_gpuit (xfc->origin, xfc->spacing, xfc->dim, 0);

    switch (xfc->xf_out_type) {
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
	if (xfc->grid_spac[0] <= 0.0f) {
	    if (xfc->xf_in->m_type == XFORM_GPUIT_BSPLINE 
		|| xfc->xf_in->m_type == XFORM_ITK_BSPLINE)
	    {
		/* Use grid spacing of input bspline */
		if (xfc->nobulk) {
		    xform_to_itk_bsp_nobulk (xfc->xf_out, xfc->xf_in, &pih, 0);
		} else {
		    xform_to_itk_bsp (xfc->xf_out, xfc->xf_in, &pih, 0);
		}
	    } else {
		print_and_exit ("Sorry, grid spacing cannot be zero\n");
	    }
	} else {
	    if (xfc->nobulk) {
		xform_to_itk_bsp_nobulk (xfc->xf_out, xfc->xf_in, &pih, 
		    xfc->grid_spac);
	    } else {
		xform_to_itk_bsp (xfc->xf_out, xfc->xf_in, &pih, 
		    xfc->grid_spac);
	    }
	}
	break;
    case XFORM_ITK_TPS:
	print_and_exit ("Sorry, couldn't convert to XFORM_ITK_TPS\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	printf ("Converting to (itk) vector field\n");
	xform_to_itk_vf (xfc->xf_out, xfc->xf_in, &pih);
	break;
    case XFORM_GPUIT_BSPLINE:
	if (xfc->grid_spac[0] <=0.0f) {
	    if (xfc->xf_in->m_type == XFORM_GPUIT_BSPLINE 
		|| xfc->xf_in->m_type == XFORM_ITK_BSPLINE)
	    {
		xform_to_gpuit_bsp (xfc->xf_out, xfc->xf_in, &pih, 0);
	    } else {
		print_and_exit ("Sorry, grid spacing cannot be zero for conversion to gpuit_bsp\n");
	    }
	} else {
	    xform_to_gpuit_bsp (xfc->xf_out, xfc->xf_in, &pih, 
		xfc->grid_spac);
	}
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
    default:
	print_and_exit ("Sorry, couldn't convert to xform (type = %d)\n",
	    xfc->xf_out_type);
	break;
    }
}
