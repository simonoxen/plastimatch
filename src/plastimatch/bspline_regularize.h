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
    Bspline_parms *parms, 
    Bspline_state *bst, 
    Bspline_xform *bxf, 
    Volume *fixed, 
    Volume *moving
);

#if defined __cplusplus
}
#endif

#endif
