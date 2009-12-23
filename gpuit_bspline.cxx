/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "bspline.h"
#include "logfile.h"
#include "mathutil.h"
#include "plm_image_header.h"
#include "plm_registration.h"
#include "readmha.h"
#include "volume.h"
#include "xform.h"

static void
do_gpuit_bspline_stage_internal (
    Registration_Parms* regp, 
    Registration_Data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_Parms* stage)
{
    BSPLINE_Parms parms;
    PlmImageHeader pih;

    logfile_printf ("Converting fixed\n");
    Volume *fixed = regd->fixed_image->gpuit_float();
    logfile_printf ("Converting moving\n");
    Volume *moving = regd->moving_image->gpuit_float();
    logfile_printf ("Done.\n");

    Volume *moving_ss, *fixed_ss;
    Volume *moving_grad = 0;

    /* Confirm grid method.  This should go away? */
    if (stage->grid_method != 1) {
	logfile_printf ("Sorry, GPUIT B-Splines must use grid method #1\n");
	exit (-1);
    }

    /* Note: Image subregion registration not yet supported */

    /* Convert images to gpuit format */
    volume_convert_to_float (moving);		    /* Maybe not necessary? */
    volume_convert_to_float (fixed);		    /* Maybe not necessary? */

    /* Subsample images */
    printf ("SUBSAMPLE: (%d %d %d), (%d %d %d)\n", 
	stage->fixed_subsample_rate[0], stage->fixed_subsample_rate[1], 
	stage->fixed_subsample_rate[2], stage->moving_subsample_rate[0], 
	stage->moving_subsample_rate[1], stage->moving_subsample_rate[2]
    );
    moving_ss = volume_subsample (moving, stage->moving_subsample_rate);
    fixed_ss = volume_subsample (fixed, stage->fixed_subsample_rate);

    logfile_printf ("moving_ss size = %d %d %d\n", moving_ss->dim[0], 
	moving_ss->dim[1], moving_ss->dim[2]);
    logfile_printf ("fixed_ss size = %d %d %d\n", fixed_ss->dim[0], 
	fixed_ss->dim[1], fixed_ss->dim[2]);

    /* Make spatial gradient image */
    moving_grad = volume_make_gradient (moving_ss);

    /* Initialize parms */
    bspline_parms_set_default (&parms);
    if (stage->optim_type == OPTIMIZATION_STEEPEST) {
	parms.optimization = BOPT_STEEPEST;
    } else {
	parms.optimization = BOPT_LBFGSB;
    }
    switch (stage->metric_type) {
    case METRIC_MSE:
	parms.metric = BMET_MSE;
	break;
    case METRIC_MI:
    case METRIC_MI_MATTES:
	parms.metric = BMET_MI;
	break;
    default:
	print_and_exit ("Undefined metric type in gpuit_bspline\n");
    }
    switch (stage->threading_type) {
    case THREADING_SINGLE:
	if (stage->alg_flavor == 0) {
	    parms.implementation = 'h';
	} else {
	    parms.implementation = stage->alg_flavor;
	}
	parms.threading = BTHR_CPU;
	break;
    case THREADING_OPENMP:
	if (stage->alg_flavor == 0) {
	    parms.implementation = 'g';
	} else {
	    parms.implementation = stage->alg_flavor;
	}
	parms.threading = BTHR_CPU;
	break;
    case THREADING_BROOK:
	/* Brook B-spline doesn't exist.  Use cuda instead. */
    case THREADING_CUDA:
	if (stage->alg_flavor == 0) {
	    parms.implementation = 'j';
	} else {
	    parms.implementation = stage->alg_flavor;
	}
	parms.threading = BTHR_CUDA;
	break;
    default:
	print_and_exit ("Undefined impl type in gpuit_bspline\n");
    }
    parms.max_its = stage->max_its;
    parms.mi_hist.fixed.bins = stage->mi_histogram_bins;
    parms.mi_hist.moving.bins = stage->mi_histogram_bins;

    /* Transform input xform to gpuit vector field */
    pih.set_from_gpuit (fixed_ss->offset, fixed_ss->pix_spacing, 
	fixed_ss->dim, fixed_ss->direction_cosines);
    xform_to_gpuit_bsp (xf_out, xf_in, &pih, stage->grid_spac);

    /* Run bspline optimization */
    bspline_optimize (xf_out->get_gpuit_bsp(), 0, &parms, fixed_ss, 
	moving_ss, moving_grad);

    /* Free up temporary memory */
    volume_free (fixed_ss);
    volume_free (moving_ss);
    volume_free (moving_grad);
    bspline_parms_free (&parms);
}

void
do_gpuit_bspline_stage (Registration_Parms* regp, 
			Registration_Data* regd, 
			 Xform *xf_out, 
			 Xform *xf_in,
			 Stage_Parms* stage)
{
    do_gpuit_bspline_stage_internal (regp, regd, xf_out, xf_in, stage);
//    printf ("Deformation stats (out)\n");
//    deformation_stats (xf_out->get_itk_vf());
}
