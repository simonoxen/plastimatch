/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _native_translation_h_
#define _native_translation_h_

class Registration_data;
class Xform;
class Stage_parms;

void
native_translation_stage (
    Registration_data* regd,
    Xform *xf_out,
    Xform *xf_in,
    Stage_parms* stage);

#endif
