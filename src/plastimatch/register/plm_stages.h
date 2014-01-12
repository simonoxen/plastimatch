/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_stages_h_
#define _plm_stages_h_

#include "plmregister_config.h"
#include "xform.h"

class Registration_data;
class Registration_parms;

PLMREGISTER_API Xform::Pointer do_registration_pure (
    Registration_data* regd,
    Registration_parms* regp
);
PLMREGISTER_API void do_registration (Registration_parms* regp);

#endif
