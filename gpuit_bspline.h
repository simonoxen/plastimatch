/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gpuit_bspline_h_
#define _gpuit_bspline_h_

class Registration_Data;
class Xform;
class Stage_Parms;

void
do_gpuit_bspline_stage (Registration_Data* regd,
			 Xform *xf_out,
			 Xform *xf_in,
			 Stage_Parms* stage);

#endif
