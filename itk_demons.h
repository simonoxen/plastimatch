/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demons_h_
#define _demons_h_

//#include "xform.h"

class Registration_Data;
class Xform;
class Stage_Parms;

void do_demons_stage (Registration_Data* regd, Xform *xf_out, Xform *xf_in, Stage_Parms* stage);

#endif
