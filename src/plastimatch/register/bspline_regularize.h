/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_h_
#define _bspline_regularize_h_

#include "plm_config.h"
#include "plmbase.h"
//#include "bspline_xform.h"

class Bspline_score;
class Bspline_regularize_state;

class Reg_parms
{
public:
    char implementation;    /* Implementation: a, b, c, etc */
    float lambda;           /* Smoothness weighting factor  */
public:
    Reg_parms () {
        this->implementation = '\0';
        this->lambda = 0.0f;
    }
};

gpuit_EXPORT
void
bspline_regularize_initialize (
    Reg_parms* reg_parms,
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize_destroy (
    Reg_parms* reg_parms,
    Bspline_regularize_state* rst,
    Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize (
    Bspline_score* bsp_score,    /* Gets updated */
    Bspline_regularize_state* rst,
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf
);

#endif
