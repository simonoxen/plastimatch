/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mse_h_
#define _bspline_mse_h_

#include "plmregister_config.h"

class Bspline_optimize_data;

PLMREGISTER_C_API void bspline_score_c_mse (Bspline_optimize_data *bod);
PLMREGISTER_C_API void bspline_score_g_mse (Bspline_optimize_data *bod);
PLMREGISTER_C_API void bspline_score_h_mse (Bspline_optimize_data *bod);
PLMREGISTER_C_API void bspline_score_i_mse (Bspline_optimize_data *bod);

#endif
