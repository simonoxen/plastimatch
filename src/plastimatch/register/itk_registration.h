/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_registration_h_
#define _itk_registration_h_

#include "plmregister_config.h"

class Registration_data;
class Stage_parms;
class Xform;

void itk_registration_stage (Registration_data* regd, Xform *xf_out, 
    Xform *xf_in, Stage_parms* stage);

#endif
