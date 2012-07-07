/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_registration_private_h_
#define _itk_registration_private_h_

#include "plmregister_config.h"
#include "itkImageRegistrationMethod.h"

class Registration_data;
class Stage_parms;
class Xform;
typedef itk::ImageRegistrationMethod < 
    FloatImageType, FloatImageType > RegistrationType;

class Itk_registration_private {
public:
    Itk_registration_private (
        Registration_data* regd, 
        Xform *xf_out, 
        Xform *xf_in, 
        Stage_parms* stage
    )
    {
        this->regd = regd;
        this->xf_in = xf_in;
        this->xf_out = xf_out;
        this->stage = stage;
    }
    ~Itk_registration_private () {}

public:
    Registration_data *regd;
    Xform *xf_out;
    Xform *xf_in;
    Stage_parms *stage;

    RegistrationType::Pointer registration;

public:
    void set_fixed_image_region ();
    void set_mask_images ();
    void set_metric ();
    void set_observer ();
    void set_optimization ();
    void set_transform ();
    void set_xf_out ();
    void show_stats ();
};

void
set_optimization (RegistrationType::Pointer registration,
		  Stage_parms* stage);

const itk::Array<double>&
optimizer_get_current_position (RegistrationType::Pointer registration, 
		  Stage_parms* stage);
int
optimizer_get_current_iteration (RegistrationType::Pointer registration, 
				 Stage_parms* stage);
double
optimizer_get_value (RegistrationType::Pointer registration, 
		     Stage_parms* stage);
double
optimizer_get_step_length (RegistrationType::Pointer registration, 
		           Stage_parms* stage);

void
optimizer_update_settings (RegistrationType::Pointer registration, 
			    Stage_parms* stage);

void
optimizer_set_max_iterations (RegistrationType::Pointer registration, 
				Stage_parms* stage, int its);

#endif
