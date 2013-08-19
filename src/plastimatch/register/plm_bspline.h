/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gpuit_bspline_h_
#define _gpuit_bspline_h_

#include "plmregister_config.h"

class Plm_bspline_private;
class Registration_parms;
class Registration_data;
class Xform;
class Stage_parms;
class Volume;

class PLMREGISTER_API Plm_bspline {
public:
    Plm_bspline_private *d_ptr;
public:
    Plm_bspline (
        Registration_parms *regp, 
        Registration_data *regd, 
        Stage_parms *stage, 
        Xform *xf_in);
    ~Plm_bspline ();

public:
    void run_stage ();
protected:
    void initialize ();
    void cleanup ();
};

void
do_gpuit_bspline_stage (
    Registration_parms* regp, 
    Registration_data* regd, 
    Xform *xf_out, 
    Xform *xf_in, 
    Stage_parms* stage);

#endif
