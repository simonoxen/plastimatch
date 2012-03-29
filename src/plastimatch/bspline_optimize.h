/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_h_
#define _bspline_optimize_h_

#include "plm_config.h"
#include "bspline.h"
#include "volume.h"


#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void bspline_optimize (
    Bspline_xform* bxf, 
    Bspline_state **bst,
    Bspline_parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad);

#if defined __cplusplus
}
#endif

#endif
