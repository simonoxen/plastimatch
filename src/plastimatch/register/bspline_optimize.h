/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_h_
#define _bspline_optimize_h_

#include "plmregister_config.h"

class Bspline_parms;
class Bspline_state;
class Bspline_xform;

PLMREGISTER_C_API void bspline_optimize (
    Bspline_xform* bxf, 
    Bspline_state **bst,
    Bspline_parms *parms);

#endif
