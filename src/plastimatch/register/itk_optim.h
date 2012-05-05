/* =======================================================================*
   Copyright (c) 2004-2006 Massachusetts General Hospital.
   All rights reserved.
 * =======================================================================*/
#ifndef _itk_optim_h_
#define _itk_optim_h_

#include "plmregister_config.h"
#include "itk_registration.h"
#include "plm_parms.h"

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
