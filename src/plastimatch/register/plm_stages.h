/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_stages_h_
#define _plm_stages_h_

#include "plm_config.h"
#include "plm_parms.h"
#include "registration_data.h"
#include "xform.h"

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */

plastimatch1_EXPORT
void
do_registration_pure (
    Xform** xf_result,
    Registration_data* regd,
    Registration_parms* regp
);

plastimatch1_EXPORT
void
do_registration (Registration_parms* regp);

#endif
