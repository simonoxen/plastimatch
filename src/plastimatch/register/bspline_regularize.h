/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_h_
#define _bspline_regularize_h_

#include "plmregister_config.h"
#include "volume.h"

class Bspline_regularize_private;
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

class PLMREGISTER_API Bspline_regularize {
public:
    SMART_POINTER_SUPPORT (Bspline_regularize);
    Bspline_regularize_private *d_ptr;
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
    void initialize (
        Reg_parms* reg_parms,
        Bspline_xform* bxf
    );
    void destroy (
        Reg_parms* reg_parms,
        Bspline_xform* bxf
    );
    void compute_score (
        Bspline_score* bsp_score,    /* Gets updated */
        const Reg_parms* reg_parms,
        const Bspline_xform* bxf
    );

public:
    void vf_regularize_analytic_init (
        const Bspline_xform* bxf);
    void vf_regularize_analytic_destroy ();
};

#endif
