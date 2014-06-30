/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_h_
#define _bspline_optimize_h_

#include "plmregister_config.h"

class Bspline_parms;
class Bspline_state;
class Bspline_xform;

class Bspline_optimize {
public:
    Bspline_xform *bxf;
    Bspline_state *bst;
    Bspline_parms *parms;
    Volume *fixed;
    Volume *moving;
    Volume *moving_grad;
public:
    Bspline_optimize () {
        bxf = 0;
        bst = 0;
        parms = 0;
        fixed = 0;
        moving = 0;
        moving_grad = 0;
    }
};

PLMREGISTER_C_API void bspline_optimize (
    Bspline_xform* bxf, 
    Bspline_state **bst,
    Bspline_parms *parms);

#endif
