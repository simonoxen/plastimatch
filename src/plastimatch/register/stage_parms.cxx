/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "shared_parms.h"
#include "stage_parms.h"

class Stage_parms_private
{
public:
    Shared_parms *shared;
public:
    Stage_parms_private () {
        shared = new Shared_parms;
    }
    Stage_parms_private (const Stage_parms_private& s) {
        this->shared = new Shared_parms (*s.shared);
    }
    ~Stage_parms_private () {
        delete shared;
    }
};

Stage_parms::Stage_parms ()
{
    d_ptr = new Stage_parms_private;

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
    /* Output files */
    img_out_fmt = IMG_OUT_FMT_AUTO;
    img_out_type = PLM_IMG_TYPE_UNDEFINED;
    *img_out_fn = 0;
    xf_out_itk = false;
    *vf_out_fn = 0;
}

Stage_parms::Stage_parms (const Stage_parms& s) 
{
    d_ptr = new Stage_parms_private (*s.d_ptr);

    /* Copy most of the parameters ... */

    /* Stage # */
    stage_no = s.stage_no;
    /* Generic optimization parms */
    xform_type = s.xform_type;
    optim_type = s.optim_type;
    impl_type = s.impl_type;
    alg_flavor = s.alg_flavor;
    threading_type = s.threading_type;
    metric_type = s.metric_type;
    regularization_type = s.regularization_type;
    regularization_lambda = s.regularization_lambda;
    /* Image subsampling */
    subsampling_type = s.subsampling_type;
    fixed_subsample_rate[0] = s.fixed_subsample_rate[0];
    fixed_subsample_rate[1] = s.fixed_subsample_rate[1];
    fixed_subsample_rate[2] = s.fixed_subsample_rate[2];
    moving_subsample_rate[0] = s.moving_subsample_rate[0];
    moving_subsample_rate[1] = s.moving_subsample_rate[1];
    moving_subsample_rate[2] = s.moving_subsample_rate[2];
    /* Intensity values for air */
    background_max = s.background_max;
    default_value = s.default_value;
    /* Generic optimization parms */
    min_its = s.min_its;
    max_its = s.max_its;
    convergence_tol = s.convergence_tol;
    /* LBGFG optimizer */
    grad_tol = s.grad_tol;
    /* LBGFGB optimizer */
    pgtol = s.pgtol;
    /* Versor & RSG optimizer */
    max_step = s.max_step;
    min_step = s.min_step;
    rsg_grad_tol = s.rsg_grad_tol;
    translation_scale_factor = s.translation_scale_factor;
    /* Quaternion optimizer */
    learn_rate = s.learn_rate;
    /* Mattes mutual information */
    mi_histogram_bins_fixed = s.mi_histogram_bins_fixed;
    mi_histogram_bins_moving = s.mi_histogram_bins_moving;
    mi_num_spatial_samples = s.mi_num_spatial_samples;
    mi_num_spatial_samples_pct = s.mi_num_spatial_samples_pct;
    mi_histogram_type = s.mi_histogram_type;
    /*Setting values to zero by default. In this case minVal and maxVal will be calculated from image*/
    mi_fixed_image_minVal = s.mi_fixed_image_minVal;
    mi_fixed_image_maxVal = s.mi_fixed_image_maxVal;
    mi_moving_image_minVal = s.mi_moving_image_minVal;
    mi_moving_image_maxVal = s.mi_moving_image_maxVal;
    /* ITK & GPUIT demons */
    demons_std = s.demons_std;
    /* GPUIT demons */
    demons_acceleration = s.demons_acceleration;
    demons_homogenization = s.demons_homogenization;
    demons_filter_width[0] = s.demons_filter_width[0];
    demons_filter_width[1] = s.demons_filter_width[1];
    demons_filter_width[2] = s.demons_filter_width[2];
    /* ITK amoeba */
    amoeba_parameter_tol = s.amoeba_parameter_tol;
    /* Bspline parms */
    num_grid[0] = s.num_grid[0];
    num_grid[1] = s.num_grid[1];
    num_grid[2] = s.num_grid[2];
    grid_spac[0] = s.grid_spac[0];
    grid_spac[1] = s.grid_spac[1];
    grid_spac[2] = s.grid_spac[2];
    grid_method = s.grid_method;
    histoeq = s.histoeq;
    /* Landmarks */
    landmark_stiffness = s.landmark_stiffness;
    landmark_flavor = s.landmark_flavor;
    /* Output files */
    img_out_type = s.img_out_type;
    xf_out_itk = s.xf_out_itk;

    /* ...but not the output filenames */
    img_out_fmt = IMG_OUT_FMT_AUTO;
    *img_out_fn = 0;
    xf_out_fn.clear ();
    *vf_out_fn = 0;

    /* ...and don't to resume unless specifically requested */
    resume_stage = false;
}

Stage_parms::~Stage_parms ()
{
    delete d_ptr;
}

Shared_parms*
Stage_parms::get_shared_parms () 
{
    return d_ptr->shared;
}

const Shared_parms*
Stage_parms::get_shared_parms () const
{
    return d_ptr->shared;
}