/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_analytic_h_
#define _bspline_regularize_analytic_h_

#include "plm_config.h"
#include "bspline_xform.h"
#include "volume.h"

class Bspline_regularize_state;
class Bspline_score;

gpuit_EXPORT
Volume*
compute_vf_from_coeff (const Bspline_xform* bxf);

gpuit_EXPORT
void
compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol);

gpuit_EXPORT
void
vf_regularize_analytic_init (
    Bspline_regularize_state* rst,
    const Bspline_xform* bxf
);

gpuit_EXPORT
void
vf_regularize_analytic_destroy (
    Bspline_regularize_state* rst
);

gpuit_EXPORT
void
vf_regularize_analytic_omp (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Bspline_regularize_state* rst,
    const Bspline_xform* bxf
);

gpuit_EXPORT
void
vf_regularize_analytic (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Bspline_regularize_state* rst,
    const Bspline_xform* bxf
);

#endif /* _reg_h_ */
