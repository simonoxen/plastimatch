/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include "plmbase_config.h"

#include "plmbase.h"
#include "plmsys.h"

#include "plm_image_header.h"
#include "itk_image.h"
#include "xform_convert.h"

void 
xform_convert (Xform_convert *xfc)
{
    Plm_image_header pih;
    pih.set_from_volume_header (xfc->m_volume_header);

    switch (xfc->m_xf_out_type) {
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
	if (xfc->m_grid_spac[0] <= 0.0f) {
	    if (xfc->m_xf_in->m_type == XFORM_GPUIT_BSPLINE 
		|| xfc->m_xf_in->m_type == XFORM_ITK_BSPLINE)
	    {
		/* Use grid spacing of input bspline */
		if (xfc->m_nobulk) {
		    xform_to_itk_bsp_nobulk (xfc->m_xf_out, 
			xfc->m_xf_in, &pih, 0);
		} else {
		    printf ("Standard case.\n");
		    pih.print ();
		    xform_to_itk_bsp (xfc->m_xf_out, xfc->m_xf_in, &pih, 0);
		}
	    } else {
		print_and_exit ("Sorry, grid spacing cannot be zero\n");
	    }
	} else {
	    if (xfc->m_nobulk) {
		xform_to_itk_bsp_nobulk (xfc->m_xf_out, xfc->m_xf_in, &pih, 
		    xfc->m_grid_spac);
	    } else {
		xform_to_itk_bsp (xfc->m_xf_out, xfc->m_xf_in, &pih, 
		    xfc->m_grid_spac);
	    }
	}
	break;
    case XFORM_ITK_TPS:
	print_and_exit ("Sorry, couldn't convert to XFORM_ITK_TPS\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	printf ("Converting to (itk) vector field\n");
	xform_to_itk_vf (xfc->m_xf_out, xfc->m_xf_in, &pih);
	break;
    case XFORM_GPUIT_BSPLINE:
	if (xfc->m_grid_spac[0] <=0.0f) {
	    if (xfc->m_xf_in->m_type == XFORM_GPUIT_BSPLINE 
		|| xfc->m_xf_in->m_type == XFORM_ITK_BSPLINE)
	    {
		xform_to_gpuit_bsp (xfc->m_xf_out, xfc->m_xf_in, &pih, 0);
	    } else {
		print_and_exit ("Sorry, grid spacing cannot be zero for conversion to gpuit_bsp\n");
	    }
	} else {
	    xform_to_gpuit_bsp (xfc->m_xf_out, xfc->m_xf_in, &pih, 
		xfc->m_grid_spac);
	}
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
    default:
	print_and_exit ("Sorry, couldn't convert to xform (type = %d)\n",
	    xfc->m_xf_out_type);
	break;
    }
}
