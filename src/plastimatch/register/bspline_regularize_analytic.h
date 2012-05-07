/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_analytic_h_
#define _bspline_regularize_analytic_h_

#include "plmregister_config.h"

class Bspline_regularize_state;
class Bspline_score;
class Bspline_xform;
class Volume;

API Volume* compute_vf_from_coeff (const Bspline_xform* bxf);
API void compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol);
API void vf_regularize_analytic_init (
    Bspline_regularize_state* rst,
    const Bspline_xform* bxf
);
API void vf_regularize_analytic_destroy (Bspline_regularize_state* rst);
API void vf_regularize_analytic_omp (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Bspline_regularize_state* rst,
    const Bspline_xform* bxf
);
API void vf_regularize_analytic (
    Bspline_score *bspline_score, 
    const Reg_parms* reg_parms,
    const Bspline_regularize_state* rst,
    const Bspline_xform* bxf
);

#endif
