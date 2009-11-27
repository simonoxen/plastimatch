/* =======================================================================*
   Copyright (c) 2004-2006 Massachusetts General Hospital.
   All rights reserved.
 * =======================================================================*/
#ifndef _itk_optim_h_
#define _itk_optim_h_

#include "plm_config.h"
#include "itk_registration.h"
#include "plm_registration.h"

void
set_optimization (RegistrationType::Pointer registration,
		  Stage_Parms* stage);

const itk::Array<double>&
optimizer_get_current_position (RegistrationType::Pointer registration, 
		  Stage_Parms* stage);
int
optimizer_get_current_iteration (RegistrationType::Pointer registration, 
				 Stage_Parms* stage);
double
optimizer_get_value (RegistrationType::Pointer registration, 
		     Stage_Parms* stage);
double
optimizer_get_step_length (RegistrationType::Pointer registration, 
		           Stage_Parms* stage);

void
optimizer_update_settings (RegistrationType::Pointer registration, 
			    Stage_Parms* stage);

void
optimizer_set_max_iterations (RegistrationType::Pointer registration, 
				Stage_Parms* stage, int its);


#endif
