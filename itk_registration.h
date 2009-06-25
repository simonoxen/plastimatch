/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_registration_h_
#define _itk_registration_h_

class Xform;

void do_itk_registration_stage (Registration_Data* regd, Xform *xf_out, Xform *xf_in, Stage_Parms* stage);
void do_itk_center_stage (Registration_Data* regd, Xform *xf_out, Xform *xf_in, Stage_Parms* stage);

//void align_center (RegistrationType::Pointer registration, Xform* xf_out, Stage_Parms* stage);

#endif
