/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_bspline_h_
#define _plm_bspline_h_

#include "plmregister_config.h"
#include "xform.h"

class Plm_bspline_private;
class Registration_parms;
class Registration_data;
class Stage_parms;
class Volume;

class PLMREGISTER_API Plm_bspline {
public:
    Plm_bspline_private *d_ptr;
public:
    Plm_bspline (
        Registration_parms *regp, 
        Registration_data *regd, 
        const Stage_parms *stage, 
        Xform *xf_in);
    ~Plm_bspline ();

public:
    void run_stage ();
protected:
    void initialize ();
    void cleanup ();
};

Xform::Pointer
do_gpuit_bspline_stage (
    Registration_parms* regp, 
    Registration_data* regd, 
    const Xform::Pointer& xf_in, 
    const Stage_parms* stage);

#endif
