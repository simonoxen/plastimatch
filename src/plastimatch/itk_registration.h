/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_registration_h_
#define _itk_registration_h_

#include "plm_config.h"
#include "itkImageRegistrationMethod.h"
#include "itk_image.h"
#include "plm_stages.h"

class Xform;

typedef itk::ImageRegistrationMethod < 
    FloatImageType, FloatImageType > RegistrationType;

void do_itk_registration_stage (Registration_Data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);
void do_itk_center_stage (Registration_Data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);

#endif
