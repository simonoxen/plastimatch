/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _stage_parms_h_
#define _stage_parms_h_

#include "plmregister_config.h"
#include <list>
#include <string>
#include <ctype.h>
#include <stdlib.h>
#include "bspline.h"            /* for enums */
#include "bspline_mi_hist.h"    /* for enums */
#include "plm_image_type.h"
#include "plm_path.h"
#include "pstring.h"
#include "threading.h"

#define STAGE_TRANSFORM_NONE                0
#define STAGE_TRANSFORM_TRANSLATION         1
#define STAGE_TRANSFORM_VERSOR              2
#define STAGE_TRANSFORM_QUATERNION          3
#define STAGE_TRANSFORM_AFFINE              4
#define STAGE_TRANSFORM_BSPLINE             5
#define STAGE_TRANSFORM_VECTOR_FIELD        6
#define STAGE_TRANSFORM_ALIGN_CENTER        7

#define OPTIMIZATION_NO_REGISTRATION        0
#define OPTIMIZATION_AMOEBA                 1
#define OPTIMIZATION_RSG                    2
#define OPTIMIZATION_VERSOR                 3
#define OPTIMIZATION_LBFGS                  4
#define OPTIMIZATION_LBFGSB                 5
#define OPTIMIZATION_DEMONS                 6
#define OPTIMIZATION_STEEPEST               7
#define OPTIMIZATION_QUAT                   8
#define OPTIMIZATION_LIBLBFGS               9
#define OPTIMIZATION_ONEPLUSONE            10
#define OPTIMIZATION_FRPR                  11

#define IMPLEMENTATION_NONE                 0
#define IMPLEMENTATION_ITK                  1
#define IMPLEMENTATION_PLASTIMATCH          2

#define OPTIMIZATION_SUB_FSF                0
#define OPTIMIZATION_SUB_DIFF_ITK           1
#define OPTIMIZATION_SUB_LOGDOM_ITK         2
#define OPTIMIZATION_SUB_SYM_LOGDOM_ITK     3

#define METRIC_NONE                         0
#define METRIC_MSE                          1
#define METRIC_MI                           2
#define METRIC_MI_MATTES                    3

#define IMG_OUT_FMT_AUTO                    0
#define IMG_OUT_FMT_DICOM                   1


enum Subsampling_type {
    SUBSAMPLING_AUTO,
    SUBSAMPLING_VOXEL_RATE
};

enum Regularization_type {
    REGULARIZATION_NONE, 
    REGULARIZATION_BSPLINE_ANALYTIC, 
    REGULARIZATION_BSPLINE_SEMI_ANALYTIC, 
    REGULARIZATION_BSPLINE_NUMERIC
};

enum Demons_gradient_type {
    SYMMETRIC,
    FIXED_IMAGE,
    WARPED_MOVING,
    MAPPED_MOVING
};

class Plm_image;
class Shared_parms;
class Stage_parms_private;

class PLMREGISTER_API Stage_parms {
public:
    Stage_parms_private *d_ptr;
public:
    Stage_parms ();
    Stage_parms (const Stage_parms& s);
    ~Stage_parms ();
public:
    /* Stage # */
    int stage_no;
    /* Stage resume? */
    bool resume_stage;
    /* Generic optimization parms */
    int xform_type;
    int optim_type;
    int impl_type;
    // sub_type used for different types of demons reg (e.g. non-diffeomorphic, diffeomorphic)
    int optim_subtype;
    char alg_flavor;
    Threading threading_type;
    int metric_type;
    Regularization_type regularization_type;
    float regularization_lambda;
    /* Image subsampling */
    Subsampling_type subsampling_type;
    int fixed_subsample_rate[3];   /* In voxels */
    int moving_subsample_rate[3];  /* In voxels */
    /* Intensity values for air */
    float background_max;          /* Threshold to find the valid region */
    float default_value;           /* Replacement when out-of-view */
    /* Generic optimization parms */
    int min_its;
    int max_its;
    float convergence_tol;
    /* LBGFG optimizer */
    float grad_tol;
    /* LBGFGB optimizer */
    float pgtol;
    /* Versor & RSG optimizer */
    float max_step;
    float min_step;
    float rsg_grad_tol;
    int translation_scale_factor;
    /*OnePlusOne evvolutionary optimizer*/

    float opo_epsilon;
    float opo_initial_search_rad;

    /*FRPR optimizer*/
    float frpr_step_tol;
    float frpr_step_length;
    int frpr_max_line_its;
    /* Quaternion optimizer */
    float learn_rate;
    /* Mutual information */
    int mi_histogram_bins_fixed;
    int mi_histogram_bins_moving;
    int mi_num_spatial_samples;
    float mi_num_spatial_samples_pct;

    enum Bspline_mi_hist_type mi_histogram_type;
    float mi_fixed_image_minVal;
    float mi_fixed_image_maxVal;
    float mi_moving_image_minVal;
    float mi_moving_image_maxVal;
    /* ITK (& GPUIT) demons */
    float demons_std;
    float demons_std_update_field;
    float demons_step_length;
    bool demons_smooth_update_field, demons_smooth_deformation_field;
    unsigned int num_approx_terms_log_demons;
    bool histoeq;         // histogram matching flag
    bool thresh_mean_intensity;
    unsigned int num_matching_points;
    unsigned int num_hist_levels;


    Demons_gradient_type demons_gradient_type;
    /* GPUIT demons */
    float demons_acceleration;
    float demons_homogenization;
    int demons_filter_width[3];
    /* ITK amoeba */
    float amoeba_parameter_tol;
    /* Bspline parms */
    int num_grid[3];     // number of grid points in x,y,z directions
    float grid_spac[3];  // absolute grid spacing in mm in x,y,z directions
    int grid_method;     // num control points (0) or absolute spacing (1)
    /* Landmarks */
    float landmark_stiffness; //strength of attraction between landmarks
    char landmark_flavor;
    /* Output files */
    int img_out_fmt;
    Plm_image_type img_out_type;
    char img_out_fn[_MAX_PATH];
    bool xf_out_itk;
    std::list<std::string> xf_out_fn;
    char vf_out_fn[_MAX_PATH];
    std::string debug_dir;
    Pstring warped_landmarks_fn;
public:
    Shared_parms *get_shared_parms ();
    const Shared_parms *get_shared_parms () const;
};

#endif
