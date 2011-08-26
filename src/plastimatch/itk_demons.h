/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demons_h_
#define _demons_h_

//#include "xform.h"

class Registration_data;
class Xform;
class Stage_parms;

void do_demons_stage (Registration_data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);

#endif
