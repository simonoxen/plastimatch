/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_h_
#define _bspline_optimize_h_

#include "plm_config.h"
#include "bspline.h"
#include "volume.h"

typedef struct bspline_optimize_data Bspline_optimize_data;
struct bspline_optimize_data
{
    BSPLINE_Xform* bxf;
    Bspline_state *bst;
    BSPLINE_Parms *parms;
    Volume *fixed;
    Volume *moving;
    Volume *moving_grad;
};

#if defined __cplusplus
extern "C" {
#endif

void
bspline_optimize (
    BSPLINE_Xform* bxf, 
    Bspline_state *bst,
    BSPLINE_Parms *parms, 
    Volume *fixed, 
    Volume *moving, 
    Volume *moving_grad);

#if defined __cplusplus
}
#endif

#endif
