/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_lbfgsb_h_
#define _bspline_optimize_lbfgsb_h_

#include <stdio.h>
#include "bspline.h"
#include "bspline_optimize.h"

#if defined __cplusplus
extern "C" {
#endif
void
bspline_optimize_lbfgsb (
    Bspline_optimize_data *bod
);
#if defined __cplusplus
}
#endif

#endif
