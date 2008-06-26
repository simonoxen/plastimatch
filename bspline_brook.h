/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_brook_h_
#define _bspline_brook_h_

#include "bspline.h"

#if defined __cplusplus
extern "C" {
#endif
void 
bspline_score_on_gpu_reference(BSPLINE_Score *ssd, 
			       Volume *fixed, Volume *moving, 
			       Volume *moving_grad, 
			       BSPLINE_Data *bspd, BSPLINE_Parms *parms);
#if defined __cplusplus
}
#endif

#endif
