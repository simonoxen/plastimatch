/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_regularize_h_
#define _bspline_regularize_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

void
bspline_regularize_score (
    BSPLINE_Parms *parms, 
    Bspline_state *bst, 
    BSPLINE_Xform *bxf, 
    Volume *fixed, 
    Volume *moving
);

#if defined __cplusplus
}
#endif

#endif
