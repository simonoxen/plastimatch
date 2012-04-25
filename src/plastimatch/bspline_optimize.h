/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_h_
#define _bspline_optimize_h_

#include "plm_config.h"
#include "plmbase.h"
#include "bspline.h"



#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void bspline_optimize (
    Bspline_xform* bxf, 
    Bspline_state **bst,
    Bspline_parms *parms);

#if defined __cplusplus
}
#endif

#endif
