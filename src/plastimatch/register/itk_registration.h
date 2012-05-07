/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_registration_h_
#define _itk_registration_h_

#include "plmregister_config.h"
#include "itkImageRegistrationMethod.h"
#include "plm_stages.h"
#include "plm_parms.h"

class Registration_data;
class Xform;

typedef itk::ImageRegistrationMethod < 
    FloatImageType, FloatImageType > RegistrationType;

void do_itk_registration_stage (Registration_data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);
void do_itk_center_stage (Registration_data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);

#endif
