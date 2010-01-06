/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>

#include "plm_int.h"
#include "getopt.h"
#include "itk_image.h"
#include "itk_warp.h"
#include "plm_image_header.h"
#include "plm_warp.h"
#include "print_and_exit.h"
#include "readmha.h"
#include "volume.h"
#include "warp_main.h"
#include "xform.h"

void
warp_image_main (Warp_parms* parms)
{
    DeformationFieldType::Pointer vf = DeformationFieldType::New();
    PlmImage im_in, im_out;
    PlmImage* im_out_ptr;
    PlmImageHeader pih;
    Xform xform;

    /* Load input image */
    im_in.load_native (parms->input_fn);

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
    } else if (xform.m_type == XFORM_GPUIT_BSPLINE) {
	/* use the spacing from input bxf file */
	pih.set_from_gpuit_bspline (xform.get_gpuit_bsp());
    } else {
	/* otherwise, use the spacing of the input image */
	pih.set_from_plm_image (&im_in);
    }

    printf ("PIH is:\n");
    pih.print ();

    /* Do the warp */
    if (parms->xf_in_fn[0]) {
	plm_warp (&im_out, &vf, &xform, &pih, &im_in, parms->default_val, 
	    parms->use_itk, parms->interp_lin);
	//do_warp (&im_out, &vf, parms, &xform, &pih, &im_in);
	im_out_ptr = &im_out;
    } else {
	im_out_ptr = &im_in;
    }

    /* Save output image */
    printf ("Saving image...\n");
    switch (parms->output_format) {
    case PLM_FILE_TYPE_DICOM_DIR:
	im_out_ptr->save_short_dicom (parms->output_fn);
	break;
    default:
	im_out_ptr->save_image (parms->output_fn);
	break;
    }

    /* Save output vector field */
    if (parms->xf_in_fn[0] && parms->vf_out_fn[0]) {
	printf ("Saving vf...\n");
	itk_image_save (vf, parms->vf_out_fn);
    }
}
