/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _reg_h_
#define _reg_h_

#include "plm_config.h"
#include "volume.h"
#include "bspline_xform.h"


class Reg_parms
{
public:
    char implementation;    /* Implementation: a, b, c, etc */
    float lambda;           /* Smoothness weighting factor  */
public:
    Reg_parms () {
        /* Init */
        this->implementation = '\0';
        this->lambda = 0.0f;
    }
};


gpuit_EXPORT
void
regularize (Reg_parms* reg_parms, Bspline_xform* bxf, float* score, float* grad);

gpuit_EXPORT
Volume*
compute_vf_from_coeff (Bspline_xform* bxf);

gpuit_EXPORT
void
compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol);

gpuit_EXPORT
float
vf_regularize_numerical (Volume* vol);


#endif /* _reg_h_ */
