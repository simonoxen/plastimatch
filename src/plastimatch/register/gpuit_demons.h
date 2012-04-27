/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gpuit_demons_h_
#define _gpuit_demons_h_

class Registration_data;
class Xform;
class Stage_parms;

void
do_gpuit_demons_stage (Registration_data* regd,
			 Xform *xf_out,
			 Xform *xf_in,
			 Stage_parms* stage);

#endif
