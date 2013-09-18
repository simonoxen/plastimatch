/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_demons_h_
#define _itk_demons_h_


#include "itk_image_type.h"

class Registration_data;
class Xform;
class Stage_parms;




void do_demons_stage (Registration_data* regd, Xform *xf_out, Xform *xf_in, Stage_parms* stage);


#endif
