/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_steepest_h_
#define _bspline_optimize_steepest_h_

#include "plm_config.h"
#include "plmbase.h"
#include "bspline.h"
#include "bspline_optimize.h"

#if defined __cplusplus
extern "C" {
#endif

void
bspline_optimize_steepest (
    Bspline_optimize_data *bod
);

#if defined __cplusplus
}
#endif

#endif
