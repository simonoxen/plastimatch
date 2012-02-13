/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_parms_h_
#define _plm_parms_h_

#include "plm_config.h"
#include <list>
#include <string>
#include <ctype.h>
#include <stdlib.h>
#include "bspline.h"
#include "plm_image.h"
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

#define IMPLEMENTATION_NONE                 0
#define IMPLEMENTATION_ITK                  1
#define IMPLEMENTATION_PLASTIMATCH          2

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

class Stage_parms {
public:
    /* Stage # */
    int stage_no;
    /* Generic optimization parms */
    int xform_type;
    int optim_type;
    int impl_type;
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
    float background_val;          /* Replacement when out-of-view */
    /* Generic optimization parms */
    int min_its;
    int max_its;
    float grad_tol;
    float convergence_tol;
    /* Versor & RSG optimizer */
    float max_step;
    float min_step;
    /* Quaternion optimizer */
    float learn_rate;
    /* Mattes mutual information */
    int mi_histogram_bins_fixed;
    int mi_histogram_bins_moving;
    int mi_num_spatial_samples;
    enum BsplineHistType mi_histogram_type;
    /* ITK & GPUIT demons */
    float demons_std;
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
    int histoeq;         // histogram matching flag on (1) or off (0)
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
    Stage_parms () {
        /* Stage # */
        stage_no = -1;
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
        background_val = -1200.0;
        /* Generic optimization parms */
        min_its = 2;
        max_its = 25;
        grad_tol = 1.5;
        convergence_tol = 5.0;
        /* Versor & RSG optimizer */
        max_step = 10.0;
        min_step = 0.5;
        /* Quaternion optimizer */
        learn_rate = 0.01 ;
        /* Mattes mutual information */
        mi_histogram_bins_fixed = 20;
        mi_histogram_bins_moving = 20;
        mi_num_spatial_samples = 10000;
        mi_histogram_type = HIST_EQSP;
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
    Stage_parms (Stage_parms& s) {
        /* Copy all the parameters except the file names */
        *this = s;
        img_out_fmt = IMG_OUT_FMT_AUTO;
        *img_out_fn = 0;
        xf_out_fn.clear ();
        *vf_out_fn = 0;
    }
};

class plastimatch1_EXPORT Registration_parms {
public:
    char moving_fn[_MAX_PATH];
    char fixed_fn[_MAX_PATH];
    char moving_mask_fn[_MAX_PATH];
    char fixed_mask_fn[_MAX_PATH];
    int img_out_fmt;
    Plm_image_type img_out_type;
    char img_out_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    bool xf_out_itk;
    //char xf_out_fn[_MAX_PATH];
    std::list<std::string> xf_out_fn;
    Pstring warped_landmarks_fn;
    Pstring fixed_landmarks_fn;
    Pstring moving_landmarks_fn;
    char vf_out_fn[_MAX_PATH];
    char log_fn[_MAX_PATH];
    int init_type;
    double init[12];

    int num_stages;
    Stage_parms** stages;

    /* for 4D and atlas */
    char moving_dir[_MAX_PATH];
    char fixed_dir[_MAX_PATH];
    char img_out_dir[_MAX_PATH];
    char vf_out_dir[_MAX_PATH];
    char moving_jobs[255][_MAX_PATH];
    char fixed_jobs[255][_MAX_PATH];
    int job_idx;
    int num_jobs;

public:
    Registration_parms() {
        *moving_fn = 0;
        *fixed_fn = 0;
        *moving_mask_fn = 0;
        *fixed_mask_fn = 0;
        img_out_fmt = IMG_OUT_FMT_AUTO;
        img_out_type = PLM_IMG_TYPE_UNDEFINED;
        *img_out_fn = 0;
        *xf_in_fn = 0;
        xf_out_itk = false;
        *vf_out_fn = 0;
        *log_fn = 0;
        init_type = STAGE_TRANSFORM_NONE;
        num_stages = 0;
        stages = 0;
        *moving_dir = 0;
        *fixed_dir = 0;
        *img_out_dir = 0;
        *vf_out_dir = 0;
        for (int i=0; i<256; i++) {
            moving_jobs[i][0] = '\0';
            fixed_jobs[i][0] = '\0';
        }
        job_idx = 0;
        num_jobs = 1;
    }
    ~Registration_parms() {
        for (int i = 0; i < num_stages; i++) {
            delete stages[i];
        }
        free (stages);
    }
public:
    int set_command_string (const std::string& command_string);
};

void not_implemented (void);
plastimatch1_EXPORT int 
plm_parms_parse_command_file (Registration_parms* regp, 
                              const char* options_fn);
plastimatch1_EXPORT int
plm_parms_process_command_file (Registration_parms *regp, FILE *fp);

#endif
