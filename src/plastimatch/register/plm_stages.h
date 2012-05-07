/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_stages_h_
#define _plm_stages_h_

#include "plmregister_config.h"

class Registration_data;
class Registration_parms;
class Xform;

API void do_registration_pure (
    Xform** xf_result,
    Registration_data* regd,
    Registration_parms* regp
);
API void do_registration (Registration_parms* regp);

#endif
