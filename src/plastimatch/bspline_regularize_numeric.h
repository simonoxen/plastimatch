/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_numeric_h_
#define _bspline_regularize_numeric_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

void
bspline_regularize_score (
    Bspline_score *ssd, 
    Reg_parms *parms, 
    Reg_state *rst,
    Bspline_xform *bxf
);

#if defined __cplusplus
}
#endif

#endif
