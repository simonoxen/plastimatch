/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "bspline.h"
#include "bspline_landmarks.h"
#include "logfile.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_image_header.h"
#include "plm_parms.h"
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
    Bspline_parms parms;
    Plm_image_header pih;

    logfile_printf ("Converting fixed\n");
    Volume *fixed = regd->fixed_image->gpuit_float();
    logfile_printf ("Converting moving\n");
    Volume *moving = regd->moving_image->gpuit_float();
    logfile_printf ("Done.\n");

    Volume *moving_ss, *fixed_ss;
    Volume *moving_grad = 0;
	Volume *vector_field = 0;

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
    case THREADING_CPU_SINGLE:
	if (stage->alg_flavor == 0) {
	    parms.implementation = 'h';
	} else {
	    parms.implementation = stage->alg_flavor;
	}
	parms.threading = BTHR_CPU;
	break;
    case THREADING_CPU_OPENMP:
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
    parms.max_feval = stage->max_its;
    parms.mi_hist.fixed.bins = stage->mi_histogram_bins;
    parms.mi_hist.moving.bins = stage->mi_histogram_bins;
	parms.young_modulus = stage->young_modulus;

    /* Load and adjust landmarks, if needed */
    if ( stage->fixed_landmarks_fn[0] && stage->moving_landmarks_fn[0] ) {
	parms.landmark_stiffness = stage->landmark_stiffness;
	parms.landmarks = bspline_landmarks_load (
	    stage->fixed_landmarks_fn, stage->moving_landmarks_fn);
	bspline_landmarks_adjust ( parms.landmarks, fixed_ss, moving_ss );
	if ( stage->landmark_flavor == 0 ) 
		parms.landmark_implementation='a';
		else parms.landmark_implementation = stage->landmark_flavor;
	logfile_printf("Loaded %d landmarks, fix %s, mov %s\n",
		parms.landmarks->num_landmarks,
		stage->moving_landmarks_fn, stage->fixed_landmarks_fn);
	}

    /* Transform input xform to gpuit vector field */
    pih.set_from_gpuit (fixed_ss->offset, fixed_ss->pix_spacing, 
	fixed_ss->dim, fixed_ss->direction_cosines);
    xform_to_gpuit_bsp (xf_out, xf_in, &pih, stage->grid_spac);

    /* Run bspline optimization */
    bspline_run_optimization (xf_out->get_gpuit_bsp(), 0, &parms, fixed_ss, 
	moving_ss, moving_grad);

	/* Warp landmarks and write them out */
    if (   stage->fixed_landmarks_fn[0] 
		&& stage->moving_landmarks_fn[0]
		&& stage->warped_landmarks_fn[0]) {
		logfile_printf("Trying to warp landmarks, output file: %s\n",
			stage->warped_landmarks_fn);
		vector_field = volume_create (fixed_ss->dim, fixed_ss->offset, 
			fixed_ss->pix_spacing,
			PT_VF_FLOAT_INTERLEAVED, 
			fixed_ss->direction_cosines, 0);
		bspline_interpolate_vf (vector_field, xf_out->get_gpuit_bsp() );
		if (vector_field){
			bspline_landmarks_warp( vector_field, &parms, xf_out->get_gpuit_bsp(), 
								fixed_ss, moving_ss );
			bspline_landmarks_write_file( stage->warped_landmarks_fn, "warped", 
				parms.landmarks->warped_landmarks, 
				parms.landmarks->num_landmarks ); 
			volume_destroy(vector_field);
		} else 
		print_and_exit ("Could not interpolate vector field for landmark warping\n");
	}

    /* Free up temporary memory */
    volume_destroy (fixed_ss);
    volume_destroy (moving_ss);
    volume_destroy (moving_grad);
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
