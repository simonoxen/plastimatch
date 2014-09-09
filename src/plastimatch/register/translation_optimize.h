/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _translation_optimize_h_
#define _translation_optimize_h_

#include "plmregister_config.h"
#include "xform.h"

class Registration_data;
class Xform;
class Stage_parms;

Xform::Pointer
translation_stage (
    Registration_data* regd,
    const Xform::Pointer& xf_in,
    Stage_parms* stage);

#endif
