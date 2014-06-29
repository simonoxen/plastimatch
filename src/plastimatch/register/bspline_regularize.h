/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_h_
#define _bspline_regularize_h_

#include "plmregister_config.h"
#include "volume.h"

class Bspline_score;
class Bspline_xform;

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

class Bspline_regularize {
public:
    Bspline_regularize ();
    ~Bspline_regularize ();
public:
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
public:
    void vf_regularize_analytic_init (
        const Bspline_xform* bxf);
    void vf_regularize_analytic_destroy ();
};

PLMREGISTER_API void bspline_regularize_initialize (
    Reg_parms* reg_parms,
    Bspline_regularize* rst,
    Bspline_xform* bxf
);
PLMREGISTER_API void bspline_regularize_destroy (
    Reg_parms* reg_parms,
    Bspline_regularize* rst,
    Bspline_xform* bxf
);
PLMREGISTER_API void bspline_regularize (
    Bspline_score* bsp_score,    /* Gets updated */
    Bspline_regularize* rst,
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf
);

#endif
