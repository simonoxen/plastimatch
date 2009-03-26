/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_lbfgsb_h_
#define _bspline_optimize_lbfgsb_h_

#include <stdio.h>
#include "bspline.h"

#if defined __cplusplus
extern "C" {
#endif
void
bspline_optimize_lbfgsb (
		BSPLINE_Xform* bxf, 
		BSPLINE_Parms *parms, 
		Volume *fixed, 
		Volume *moving, 
		Volume *moving_grad);
#if defined __cplusplus
}
#endif

#endif
