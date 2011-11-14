/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mse_h_
#define _bspline_mse_h_

#include "plm_config.h"
#include "bspline.h"
#include "volume.h"

/* -----------------------------------------------------------------------
   Function declarations
   ----------------------------------------------------------------------- */
#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
bspline_score_c_mse (
    Bspline_parms *parms, 
    Bspline_state *bst,
    Bspline_xform* bxf, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad
);
gpuit_EXPORT
void
bspline_score_g_mse (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad);

gpuit_EXPORT
void
bspline_score_h_mse (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad);

gpuit_EXPORT
void
bspline_score_i_mse (
    Bspline_parms *parms,
    Bspline_state *bst, 
    Bspline_xform *bxf,
    Volume *fixed,
    Volume *moving,
    Volume *moving_grad);

#if defined __cplusplus
}
#endif

#endif
