/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_numeric_h_
#define _bspline_regularize_numeric_h_

#include "plmregister_config.h"

class Bspline_regularize_state;
class Bspline_xform;
class Reg_parms;
class Volume;

PLMREGISTER_API void vf_regularize_numerical (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf,
    const Volume* vol
);
PLMREGISTER_API void bspline_regularize_numeric_a_init (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);
PLMREGISTER_API void bspline_regularize_numeric_a (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform *bxf
);
PLMREGISTER_API void bspline_regularize_numeric_a_destroy (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);
PLMREGISTER_API void bspline_regularize_numeric_d_init (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);
PLMREGISTER_API void bspline_regularize_numeric_d (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf
);
PLMREGISTER_API void bspline_regularize_numeric_d_destroy (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);

#endif
