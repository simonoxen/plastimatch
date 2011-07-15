/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _reg_h_
#define _reg_h_

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
        /* Init */
        this->implementation = '\0';
        this->lambda = 0.0f;
    }
};

#if defined (commentout)
#endif

gpuit_EXPORT
void
regularize (
    Bspline_score* bsp_score,    /* Gets updated */
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf
);

gpuit_EXPORT
Volume*
compute_vf_from_coeff (const Bspline_xform* bxf);

gpuit_EXPORT
void
compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol);

gpuit_EXPORT
float
vf_regularize_numerical (Volume* vol);

gpuit_EXPORT
float
vf_regularize_analytic (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Bspline_xform* bxf);

#endif /* _reg_h_ */
