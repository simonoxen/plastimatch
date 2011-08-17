/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_h_
#define _bspline_regularize_h_

#include "plm_config.h"
#include "bspline_xform.h"
#include "volume.h"

class Bspline_score;

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


typedef struct reg_state_struct Reg_state;
struct reg_state_struct {
    /* carry volume addresses here for now */
    Volume* fixed;
    Volume* moving;

    /* analytic methods */
    double* QX_mats;    /* Three 4x4 matrices */
    double* QY_mats;    /* Three 4x4 matrices */
    double* QZ_mats;    /* Three 4x4 matrices */
    double** QX;
    double** QY;
    double** QZ;
    double* V_mats;     /* The 6 64x64 V matricies */
    double** V;
    double* cond;
};

gpuit_EXPORT
void
bspline_regularize_initialize (
    Reg_parms* reg_parms,
    Reg_state* rst,
    Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize_destroy (
    Reg_parms* reg_parms,
    Reg_state* rst,
    Bspline_xform* bxf
);

gpuit_EXPORT
void
bspline_regularize (
    Bspline_score* bsp_score,    /* Gets updated */
    Reg_state* rst,
    Reg_parms* reg_parms,
    Bspline_xform* bxf
);


#endif
