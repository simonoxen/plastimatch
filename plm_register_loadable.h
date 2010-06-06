/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_register_loadable_h_
#define _plm_register_loadable_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_registration.h"

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT
void
plm_register_loadable (
    FloatImageType::ConstPointer fixed, 
    FloatImageType::ConstPointer moving
);
#endif
