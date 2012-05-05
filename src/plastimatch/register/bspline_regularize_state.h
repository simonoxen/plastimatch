/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_state_h_
#define _bspline_regularize_state_h_

#include "plmregister_config.h"

class Volume;

class Bspline_regularize_state {
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
};

#endif
