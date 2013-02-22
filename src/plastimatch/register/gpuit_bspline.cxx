/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bspline.h"
#include "bspline_landmarks.h"
#include "bspline_optimize.h"
#include "bspline_regularize.h"
#include "bspline_interpolate.h"
#include "gpuit_bspline.h"
#include "logfile.h"
#include "plm_image_header.h"
#include "plm_image.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "registration_data.h"
#include "registration_parms.h"
#include "stage_parms.h"
#include "volume.h"
#include "volume_resample.h"
#include "xform.h"
#include "landmark_warp.h"
#include "pointset.h"
#include "raw_pointset.h"

static void
do_gpuit_bspline_stage_internal (
    Registration_parms* regp, 
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage)
{
    Bspline_parms parms;
    Plm_image_header pih;

    logfile_printf ("Converting fixed\n");
    Volume *fixed = regd->fixed_image->gpuit_float();
    logfile_printf ("Converting moving\n");
    Volume *moving = regd->moving_image->gpuit_float();
    logfile_printf ("Done.\n");

    Volume *m_mask = NULL;
    Volume *m_mask_ss = NULL;
    Volume *f_mask = NULL;
    Volume *f_mask_ss = NULL;
    Volume *moving_ss, *fixed_ss;
    Volume *moving_grad = 0;

    /* load "stage" masks */
    /*   stage mask overrides a global mask */
    if (stage->fixed_mask_fn[0]) {
        logfile_printf ("Loading fixed mask: %s\n", stage->fixed_mask_fn);
        stage->fixed_mask = plm_image_load (stage->fixed_mask_fn, PLM_IMG_TYPE_ITK_UCHAR);
        f_mask = stage->fixed_mask->gpuit_uchar();
    } else {
        stage->fixed_mask = 0;
        if (regd->fixed_mask) {
            f_mask = regd->fixed_mask->gpuit_uchar();
        }
    }

    if (stage->moving_mask_fn[0]) {
        logfile_printf ("Loading moving mask: %s\n", stage->moving_mask_fn);
        stage->moving_mask = plm_image_load (stage->moving_mask_fn, PLM_IMG_TYPE_ITK_UCHAR);
        m_mask = stage->moving_mask->gpuit_uchar();
    } else {
        stage->moving_mask = 0;
        if (regd->moving_mask) {
            m_mask = regd->moving_mask->gpuit_uchar();
        }
    }

    /* Confirm grid method.  This should go away? */
    if (stage->grid_method != 1) {
        logfile_printf ("Sorry, GPUIT B-Splines must use grid method #1\n");
        exit (-1);
    }

    /* Note: Image subregion registration not yet supported */

    /* Convert images to gpuit format */
    volume_convert_to_float (moving);               /* Maybe not necessary? */
    volume_convert_to_float (fixed);                /* Maybe not necessary? */

    /* Subsample images */
    logfile_printf ("SUBSAMPLE: (%d %d %d), (%d %d %d)\n", 
        stage->fixed_subsample_rate[0], stage->fixed_subsample_rate[1], 
        stage->fixed_subsample_rate[2], stage->moving_subsample_rate[0], 
        stage->moving_subsample_rate[1], stage->moving_subsample_rate[2]
    );
    moving_ss = volume_subsample (moving, stage->moving_subsample_rate);
    fixed_ss = volume_subsample (fixed, stage->fixed_subsample_rate);

    /* Subsample masks (if we are using them) */
    if (m_mask) {
        m_mask_ss = volume_subsample_nn (m_mask, stage->moving_subsample_rate);
    }
    if (f_mask) {
        f_mask_ss = volume_subsample_nn (f_mask, stage->fixed_subsample_rate);
    }

    logfile_printf ("moving_ss size = %d %d %d\n", moving_ss->dim[0], 
        moving_ss->dim[1], moving_ss->dim[2]);
    logfile_printf ("fixed_ss size = %d %d %d\n", fixed_ss->dim[0], 
        fixed_ss->dim[1], fixed_ss->dim[2]);

    /* Make spatial gradient image */
    moving_grad = volume_make_gradient (moving_ss);

    /* --- Initialize parms --- */

    /* Images */
    parms.fixed = fixed_ss;
    parms.moving = moving_ss;
    parms.moving_grad = moving_grad;
    if (f_mask) parms.fixed_mask = f_mask_ss;
    if (m_mask) parms.moving_mask = m_mask_ss;

    /* Optimization */
    if (stage->optim_type == OPTIMIZATION_STEEPEST) {
        parms.optimization = BOPT_STEEPEST;
    } else if (stage->optim_type == OPTIMIZATION_LIBLBFGS) {
        parms.optimization = BOPT_LIBLBFGS;
    } else {
        parms.optimization = BOPT_LBFGSB;
    }
    parms.lbfgsb_pgtol = stage->pgtol;

    /* Metric */
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

    /* Threading */
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

    /* Regularization */
    parms.reg_parms->lambda = stage->regularization_lambda;
    switch (stage->regularization_type) {
    case REGULARIZATION_NONE:
        parms.reg_parms->lambda = 0.0f;
        break;
    case REGULARIZATION_BSPLINE_ANALYTIC:
        if (stage->threading_type == THREADING_CPU_OPENMP) {
            parms.reg_parms->implementation = 'c';
        } else {
            parms.reg_parms->implementation = 'b';
        }
        break;
    case REGULARIZATION_BSPLINE_SEMI_ANALYTIC:
        parms.reg_parms->implementation = 'd';
        break;
    case REGULARIZATION_BSPLINE_NUMERIC:
        parms.reg_parms->implementation = 'a';
        break;
    default:
        print_and_exit ("Undefined regularization type in gpuit_bspline\n");
    }
    /* JAS 2011.08.17
     * This needs to be integrated with the above switch 
     * young_modulus is regularization implementation 'd' */
    if (stage->regularization_lambda != 0) {
        parms.reg_parms->lambda = stage->regularization_lambda;
    }

    logfile_printf ("Regularization: flavor = %c lambda = %f\n", 
        parms.reg_parms->implementation,
        parms.reg_parms->lambda);

    /* Mutual information histograms */
    parms.mi_hist.fixed.type  = (BsplineHistType)stage->mi_histogram_type;
    parms.mi_hist.moving.type = (BsplineHistType)stage->mi_histogram_type;
    parms.mi_hist.fixed.bins  = stage->mi_histogram_bins_fixed;
    parms.mi_hist.moving.bins = stage->mi_histogram_bins_moving;
    parms.mi_hist.joint.bins  = parms.mi_hist.fixed.bins
                              * parms.mi_hist.moving.bins;

    /* Other stuff */
    parms.max_its = stage->max_its;
    parms.max_feval = stage->max_its;

    /* Landmarks */
    if (regd->fixed_landmarks && regd->moving_landmarks) {
        logfile_printf ("Landmarks: %d fixed, %d moving, lambda = %f\n",
            regd->fixed_landmarks->count(),
            regd->moving_landmarks->count(),
            stage->landmark_stiffness);
        parms.blm->set_landmarks (regd->fixed_landmarks, 
            regd->moving_landmarks);
        parms.blm->landmark_implementation = stage->landmark_flavor;
        parms.blm->landmark_stiffness = stage->landmark_stiffness;
    }

    /* Transform input xform to gpuit vector field */
    pih.set_from_gpuit (fixed_ss->dim, 
        fixed_ss->offset, fixed_ss->spacing, 
        fixed_ss->direction_cosines);
    xform_to_gpuit_bsp (xf_out, xf_in, &pih, stage->grid_spac);

    /* Set debugging directory */
    if (stage->debug_dir != "") {
        parms.debug = 1;
        parms.debug_dir = stage->debug_dir;
        parms.debug_stage = stage->stage_no;
        logfile_printf ("Set debug directory to %s (%d)\n", 
            parms.debug_dir.c_str(), parms.debug_stage);
    }

    /* Run bspline optimization */
    bspline_optimize (xf_out->get_gpuit_bsp(), 0, &parms);

    /* Warp landmarks and write them out */

// NSh uncomment 2013-02-14
//#if defined (commentout)
 
	logfile_printf("Trying to start landmark warper\n");
	printf("stdout, Trying to start landmark warper\n");

    if (   regd->fixed_landmarks   // does not exist if landmarks are not used 
	&& regd->moving_landmarks
	&& !regd->fixed_landmarks->point_list.empty()  
        && !regd->moving_landmarks->point_list.empty() 
        && regp->warped_landmarks_fn.not_empty() ) {
        logfile_printf("Trying to warp %d landmarks, output file: %s\n",
		regd->fixed_landmarks->point_list.size(),            
		(const char*) regp->warped_landmarks_fn);
	Volume *vector_field;        
	vector_field = new Volume (fixed_ss->dim, fixed_ss->offset, 
            fixed_ss->spacing, fixed_ss->direction_cosines, 
            PT_VF_FLOAT_INTERLEAVED, 3);
        bspline_interpolate_vf (vector_field, xf_out->get_gpuit_bsp() );
        if (vector_field) {

	logfile_printf("... vector field ok\n");

 	// using Landmark_warp code from RBF
	Landmark_warp lw;
	lw.m_fixed_landmarks  = pointset_create ();
	lw.m_moving_landmarks = pointset_create ();
	
	// regd->fixed_landmarks is Pointset, lw.m_fixed_landmarks is Raw_pointset 

	for(std::vector<Labeled_point>::iterator it = 
	          regd->fixed_landmarks->point_list.begin();
	    it != regd->fixed_landmarks->point_list.end();
	    ++it
	    ) 
	{	
	float lm[3];
	lm[0]=(*it).p[0];
	lm[1]=(*it).p[1];
	lm[2]=(*it).p[2];
	pointset_add_point_noadjust( lw.m_fixed_landmarks, lm); 
	}

	for(std::vector<Labeled_point>::iterator it = 
	          regd->moving_landmarks->point_list.begin();
	    it != regd->moving_landmarks->point_list.end();
	    ++it
	    ) 
	{	
	float lm[3];
	lm[0]=(*it).p[0];
	lm[1]=(*it).p[1];
	lm[2]=(*it).p[2];
	pointset_add_point_noadjust( lw.m_moving_landmarks, lm); 
	}

	//moving_ss is Volume*, lw.m_input_img is Plm_image*

	Volume *moving_ss_clone = new Volume();
	moving_ss_clone = moving_ss->clone();
	Plm_image *moving_ss_plmimage = new Plm_image();
	moving_ss_plmimage->set_gpuit( moving_ss_clone );

//	lw.m_input_img = regd->moving_image; // Plm_image
        lw.m_input_img = moving_ss_plmimage;
	lw.m_pih.set_from_plm_image (lw.m_input_img); // see landmark_warp_main.cxx

	// how to pass vector field into lw via Xform lw->m_vf ? 
	// lw.m_vf = xf_out;
	// this results in a runtime error:
	// Typecast error in get_gpuit_vf() - in calculate_warped_landmarks(), line 200 of landmark_warp.cxx 

	// warp & save
	lw.m_warped_landmarks = pointset_create ();
	calculate_warped_landmarks_by_vf ( &lw, vector_field);
	//calculate_warped_landmarks ( &lw);
	logfile_printf("... trying to save landmarks to %s\n",regp->warped_landmarks_fn.c_str());
	pointset_save (lw.m_warped_landmarks, regp->warped_landmarks_fn.c_str());

	/*  // obsolete code using Blm 
           bspline_landmarks_warp (vector_field, &parms, 
                xf_out->get_gpuit_bsp(), fixed_ss, moving_ss );
            bspline_landmarks_write_file (
                (const char*) regp->warped_landmarks_fn, 
                "warped", 
                regp->landmarks->warped_landmarks, 
                regp->landmarks->num_landmarks);
	*/

            delete vector_field;
        } else 
            print_and_exit ("Could not interpolate vector field for landmark warping\n");
    }
//#endif

    /* Free up temporary memory */
    if (stage->fixed_mask) {
        logfile_printf ("Freeing fixed mask.\n");
        delete stage->fixed_mask;
    }
    if (stage->moving_mask) {
        logfile_printf ("Freeing moving mask.\n");
        delete stage->moving_mask;
    }

    delete fixed_ss;
    delete moving_ss;
    delete moving_grad;
    bspline_parms_free (&parms);
}

void
do_gpuit_bspline_stage (
    Registration_parms* regp, 
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in,
    Stage_parms* stage)
{
    do_gpuit_bspline_stage_internal (regp, regd, xf_out, xf_in, stage);
//    printf ("Deformation stats (out)\n");
//    deformation_stats (xf_out->get_itk_vf());
}
