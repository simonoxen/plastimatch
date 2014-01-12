/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _native_translation_h_
#define _native_translation_h_

#include "plmregister_config.h"
#include "xform.h"

class Registration_data;
class Xform;
class Stage_parms;

Xform::Pointer
native_translation_stage (
    Registration_data* regd,
    const Xform::Pointer& xf_in,
    Stage_parms* stage);

#endif
