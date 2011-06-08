/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _reg_h_
#define _reg_h_

#include "plm_config.h"
#include "volume.h"
#include "bspline_xform.h"

gpuit_EXPORT
void
compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol);

gpuit_EXPORT
float
vf_regularize_numerical (Volume* vol);

class Reg_parms
{
public:
    char implementation;    /* Implementation: a, b, c, etc */
    bool analytic;          /* Use analytic gradient?       */
public:
    Reg_parms () {
        /* Init */
        this->implementation = '\0';
        this->analytic = false;
    }
};

#endif /* _reg_h_ */
