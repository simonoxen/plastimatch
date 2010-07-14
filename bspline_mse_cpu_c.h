/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mse_cpu_c_h_
#define _bspline_mse_cpu_c_h_

#include "plm_config.h"
#include "bspline.h"
#include "bspline_macros.h"

#if defined __cplusplus
extern "C" {
#endif

void
bspline_score_c_mse (
    BSPLINE_Parms *parms, 
    Bspline_state *bst,
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
);
#if defined __cplusplus
}
#endif

#endif
