/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdlib.h>
#include <string.h>

#include "stage_parms.h"

Stage_parms::Stage_parms ()
{
    /* Stage # */
    stage_no = -1;
    /* Stage resume? */
    resume_stage = false;
    /* Generic optimization parms */
    xform_type = STAGE_TRANSFORM_VERSOR;
    optim_type = OPTIMIZATION_VERSOR;
    impl_type = IMPLEMENTATION_NONE;
    alg_flavor = 0;
    threading_type = THREADING_CPU_OPENMP;
    metric_type = METRIC_MSE;
    regularization_type = REGULARIZATION_BSPLINE_ANALYTIC;
    regularization_lambda = 0.0f;
    /* Image subsampling */
    subsampling_type = SUBSAMPLING_AUTO;
    fixed_subsample_rate[0] = 4;
    fixed_subsample_rate[1] = 4;
    fixed_subsample_rate[2] = 1;
    moving_subsample_rate[0] = 4;
    moving_subsample_rate[1] = 4;
    moving_subsample_rate[2] = 1;
    /* Intensity values for air */
    background_max = -999.0;
    default_value = 0.0;
    /* Generic optimization parms */
    min_its = 2;
    max_its = 25;
    convergence_tol = 5.0;
    /* LBGFG optimizer */
    grad_tol = 1.5;
    /* LBGFGB optimizer */
    pgtol = 1.0e-5;
    /* Versor & RSG optimizer */
    max_step = 10.0;
    min_step = 0.5;
    rsg_grad_tol = 0.0001;
    translation_scale_factor = 1000;
    /* Quaternion optimizer */
    learn_rate = 0.01 ;
    /* Mattes mutual information */
    mi_histogram_bins_fixed = 20;
    mi_histogram_bins_moving = 20;
    mi_num_spatial_samples = -1;
    mi_num_spatial_samples_pct = 0.3;
    mi_histogram_type = HIST_EQSP;
    /*Setting values to zero by default. In this case minVal and maxVal will be calculated from image*/
    mi_fixed_image_minVal=0;
    mi_fixed_image_maxVal=0;
    mi_moving_image_minVal=0;
    mi_moving_image_maxVal=0;
    /* ITK & GPUIT demons */
    demons_std = 6.0;
    /* GPUIT demons */
    demons_acceleration = 1.0;
    demons_homogenization = 1.0;
    demons_filter_width[0] = 3;
    demons_filter_width[1] = 3;
    demons_filter_width[2] = 3;
    /* ITK amoeba */
    amoeba_parameter_tol = 1.0;
    /* Bspline parms */
    num_grid[0] = 10;
    num_grid[1] = 10;
    num_grid[2] = 10;
    grid_spac[0] = 20.;
    grid_spac[1] = 20.;
    grid_spac[2] = 20.; 
    grid_method = 1;     // by default goes to the absolute spacing
    histoeq = 0;         // by default, don't do it
    /* Landmarks */
    landmark_stiffness = 1.0;
    landmark_flavor = 'a';
    /* ROI */
    fixed_roi_enable = false;
    moving_roi_enable = false;
    fixed_roi = 0;
    moving_roi = 0;
    /* Output files */
    img_out_fmt = IMG_OUT_FMT_AUTO;
    img_out_type = PLM_IMG_TYPE_UNDEFINED;
    *img_out_fn = 0;
    xf_out_itk = false;
    *vf_out_fn = 0;
}

Stage_parms::Stage_parms (const Stage_parms& s)
{
    /* Copy all the parameters */
    *this = s;
    /* ...but not the output filenames */
    img_out_fmt = IMG_OUT_FMT_AUTO;
    *img_out_fn = 0;
    xf_out_fn.clear ();
    *vf_out_fn = 0;
    /* ...and don't to resume unless specifically requested */
    resume_stage = false;
}
