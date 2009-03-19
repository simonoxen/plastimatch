/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_registration_h_
#define _plm_registration_h_

#include <stdlib.h>
#include "itkDemonsRegistrationFilter.h"
#include "itkImageRegistrationMethod.h"
#include "plm_config.h"
#include "plm_image.h"

/* Registration typedefs */
typedef itk::ImageRegistrationMethod < 
    FloatImageType, FloatImageType > RegistrationType;
typedef itk::DemonsRegistrationFilter<
                            FloatImageType,
                            FloatImageType,
                            DeformationFieldType> DemonsFilterType;


#define STAGE_TRANSFORM_NONE		    0
#define STAGE_TRANSFORM_TRANSLATION	    1
#define STAGE_TRANSFORM_VERSOR		    2
#define STAGE_TRANSFORM_AFFINE		    3
#define STAGE_TRANSFORM_BSPLINE		    4
#define STAGE_TRANSFORM_VECTOR_FIELD	    5

#define OPTIMIZATION_NO_REGISTRATION	    0
#define OPTIMIZATION_AMOEBA		    1
#define OPTIMIZATION_RSG		    2
#define OPTIMIZATION_VERSOR		    3
#define OPTIMIZATION_LBFGS		    4
#define OPTIMIZATION_LBFGSB		    5
#define OPTIMIZATION_DEMONS		    6
#define OPTIMIZATION_STEEPEST		    7

#define IMPLEMENTATION_NONE		    0
#define IMPLEMENTATION_ITK		    1
#define IMPLEMENTATION_GPUIT_CPU	    2
#define IMPLEMENTATION_GPUIT_BROOK	    3

#define METRIC_NONE			    0
#define METRIC_MSE			    1
#define METRIC_MI			    2
#define METRIC_MI_MATTES		    3

#define IMG_OUT_FMT_AUTO		    0
#define IMG_OUT_FMT_DICOM		    1

class Stage_Parms {
public:
    int xform_type;
    int optim_type;
    int impl_type;
    int metric_type;
    int resolution[3];
    float background_max;   /* This is used as a threshold to find the valid region */
    float background_val;   /* This is used for replacement when resampling */
    int min_its;
    int max_its;
    float grad_tol;
    float convergence_tol;
    float max_step;
    float min_step;
    int mi_histogram_bins;
    int mi_num_spatial_samples;
    float demons_std;
    float demons_acceleration;
    float demons_homogenization;
    int demons_filter_width[3];
    float amoeba_parameter_tol;
    int num_grid[3];     // number of grid points in x,y,z directions
    float grid_spac[3];  // absolute grid spacing in mm in x,y,z directions
    int grid_method;     // which grid method used, numbers (0) or absolute spacing (1)
    int histoeq;         // histogram matching flag on (1) or off (0)

    int img_out_fmt;
    char img_out_fn[_MAX_PATH];
    char xf_out_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];

public:
    Stage_Parms () {
	/* Generic optimization parms */
	xform_type = STAGE_TRANSFORM_VERSOR;
	optim_type = OPTIMIZATION_VERSOR;
	impl_type = IMPLEMENTATION_ITK;
	metric_type = METRIC_MSE;
	resolution[0] = 4;
	resolution[1] = 4;
	resolution[2] = 1;
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
	/* Mattes mutual information */
	mi_histogram_bins = 20;
	mi_num_spatial_samples = 10000;
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
	/* Output files */
	img_out_fmt = IMG_OUT_FMT_AUTO;
	*img_out_fn = 0;
	*xf_out_fn = 0;
	*vf_out_fn = 0;
    }
    Stage_Parms (Stage_Parms& s) {
	/* Copy all the parameters except the file names */
	*this = s;
	img_out_fmt = IMG_OUT_FMT_AUTO;
	*img_out_fn = 0;
	*xf_out_fn = 0;
	*vf_out_fn = 0;
    }
};

class Registration_Parms {
public:
    char moving_fn[_MAX_PATH];
    char fixed_fn[_MAX_PATH];
    char moving_mask_fn[_MAX_PATH];
    char fixed_mask_fn[_MAX_PATH];
    int img_out_fmt;
    char img_out_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    char xf_out_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    char log_fn[_MAX_PATH];
    int init_type;
    double init[12];

    int num_stages;
    Stage_Parms** stages;

public:
    Registration_Parms() {
	*moving_fn = 0;
	*fixed_fn = 0;
	*moving_mask_fn = 0;
	*fixed_mask_fn = 0;
	img_out_fmt = IMG_OUT_FMT_AUTO;
	*img_out_fn = 0;
	*xf_in_fn = 0;
	*xf_out_fn = 0;
	*vf_out_fn = 0;
	*log_fn = 0;
	init_type = STAGE_TRANSFORM_NONE;
	num_stages = 0;
	stages = 0;
    }
    ~Registration_Parms() {
	for (int i = 0; i < num_stages; i++) {
	    delete stages[i];
	}
	if (stages) free (stages);
    }
};

class Registration_Data {
public:
    /* Input images */
    PlmImage *fixed_image;
    PlmImage *moving_image;
    UCharImageType::Pointer fixed_mask;
    UCharImageType::Pointer moving_mask;

    /* Region of interest */
    FloatImageType::RegionType fixed_region;
    FloatImageType::PointType fixed_region_origin;
    FloatImageType::SpacingType fixed_region_spacing;
};

void not_implemented (void);
plastimatch1_EXPORT void do_registration (Registration_Parms* regp);
plastimatch1_EXPORT int parse_command_file (Registration_Parms* regp, 
					    const char* options_fn);

#endif
