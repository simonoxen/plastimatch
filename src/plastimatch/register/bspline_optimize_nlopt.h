/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_optimize_nlopt_h_
#define _bspline_optimize_nlopt_h_

#include "plmregister_config.h"
#include "bspline_optimize.h"
#if (NLOPT_FOUND)
#include "nlopt.h"
#endif

#if defined __cplusplus
extern "C" {
#endif

#if (NLOPT_FOUND)
void
bspline_optimize_nlopt (Bspline_optimize_data *bod, nlopt_algorithm algorithm);
#endif

#if defined __cplusplus
}
#endif

#endif
