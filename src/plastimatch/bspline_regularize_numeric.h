/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_numeric_h_
#define _bspline_regularize_numeric_h_

#include "plm_config.h"

class Bspline_regularize_state;

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
vf_regularize_numerical (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf,
    const Volume* vol
);

gpuit_EXPORT
void
bspline_regularize_numeric_a_init (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize_numeric_a (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform *bxf
);

gpuit_EXPORT
void
bspline_regularize_numeric_a_destroy (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);


gpuit_EXPORT
void
bspline_regularize_numeric_d_init (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize_numeric_d (
    Bspline_score *ssd, 
    const Reg_parms *parms, 
    const Bspline_regularize_state *rst,
    const Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize_numeric_d_destroy (
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);


#if defined __cplusplus
}
#endif

#endif
