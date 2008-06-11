/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "plm_config.h"
#include "rad_registration.h"
#include "xform.h"
#include "readmha.h"
#include "volume.h"
#include "bspline.h"
#include "mathutil.h"

void
do_gpuit_bspline_stage_internal (Registration_Data* regd, 
				    Xform *xf_out, 
				    Xform *xf_in, 
				    Stage_Parms* stage)
{
    int d;
    Xform_GPUIT_Bspline *xgb;
    BSPLINE_Parms *parms;
    printf ("Converting fixed\n");
    Volume *fixed = regd->fixed_image->gpuit_float();
    printf ("Converting fixed\n");
    Volume *moving = regd->moving_image->gpuit_float();
    printf ("Done.\n");
    Volume *moving_ss, *fixed_ss;
    Volume *moving_grad = 0;
    Volume *vf_out = 0;
    Volume *vf_in = 0;

    /* Confirm grid method.  This should go away? */
    if (stage->grid_method != 1) {
	printf ("Sorry, GPUIT B-Splines must use grid method #1\n");
	exit (-1);
    }

    /* Note: Image subregion registration not yet supported */

    /* Convert images to gpuit format */
    volume_convert_to_float (moving);		    /* Maybe not necessary? */
    volume_convert_to_float (fixed);		    /* Maybe not necessary? */

    /* Subsample images */
    printf ("SUBSAMPLE: %d %d %d\n", stage->resolution[0], stage->resolution[1], stage->resolution[2]);
    moving_ss = volume_subsample (moving, stage->resolution);
    fixed_ss = volume_subsample (fixed, stage->resolution);

    /* Make spatial gradient image */
    moving_grad = volume_make_gradient (moving_ss);

    /* Initialize parms */
    xgb = (Xform_GPUIT_Bspline*) malloc (sizeof(Xform_GPUIT_Bspline));
    parms = &xgb->parms;
    bspline_default_parms (parms);
    if (stage->optim_type == OPTIMIZATION_STEEPEST) {
	parms->algorithm = BA_STEEPEST;
    } else {
	parms->algorithm = BA_LBFGSB;
    }
    parms->max_its = stage->max_its;
    for (d = 0; d < 3; d++) {
	parms->vox_per_rgn[d] = ROUND_INT (stage->grid_spac[d] / fixed_ss->pix_spacing[d]);
	if (parms->vox_per_rgn[d] < 4) {
	    printf ("Warning: grid spacing too fine (%g mm) relative to pixel size (%g mm)\n",
		    stage->grid_spac[d], fixed_ss->pix_spacing[d]);
	    parms->vox_per_rgn[d] = 4;
	}
	parms->img_origin[d] = fixed_ss->offset[d];
	parms->img_spacing[d] = fixed_ss->pix_spacing[d];
	parms->img_dim[d] = fixed_ss->dim[d];
	parms->roi_offset[d] = 0;
	parms->roi_dim[d] = fixed_ss->dim[d];
	//parms->grid_spac[d] = stage->grid_spac[d];
	xgb->grid_spac[d] = parms->vox_per_rgn[d] * fixed_ss->pix_spacing[d];
	xgb->img_origin[d] = fixed_ss->offset[d];
	xgb->img_spacing[d] = fixed_ss->pix_spacing[d];
    }

    /* Allocate memory and build LUTs */
    bspline_initialize (parms);

    /* Print out some stuff for user */
    printf ("GPUIT SPACING: %d %d %d\n", parms->vox_per_rgn[0], parms->vox_per_rgn[1], 
	    parms->vox_per_rgn[2]);
    printf ("FIXED IMG PIX SPACING: %g %g %g\n", fixed_ss->pix_spacing[0], fixed_ss->pix_spacing[1],
	 fixed_ss->pix_spacing[2]);

    /* Transform input xform to gpuit vector field */
    xform_to_gpuit_bsp (xf_out, xf_in, xgb);

    /* Run bspline optimization */
    if (stage->impl_type == IMPLEMENTATION_GPUIT_CPU) {
	parms->method = BM_CPU;
	bspline_optimize (parms, fixed_ss, moving_ss, moving_grad);
    } else {
	parms->method = BM_BROOK;
	bspline_optimize (parms, fixed_ss, moving_ss, moving_grad);
    }

    /* Free up temporary memory */
    volume_free (fixed_ss);
    volume_free (moving_ss);
    volume_free (moving_grad);
}

void
do_gpuit_bspline_stage (Registration_Data* regd, 
			 Xform *xf_out, 
			 Xform *xf_in,
			 Stage_Parms* stage)
{
    do_gpuit_bspline_stage_internal (regd, xf_out, xf_in, stage);
//    printf ("Deformation stats (out)\n");
//    deformation_stats (xf_out->get_itk_vf());
}
