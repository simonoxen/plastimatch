/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mse_h_
#define _bspline_mse_h_

#include "plmregister_config.h"
#include "plmbase.h"
#include "bspline.h"
#include "bspline_optimize.h"

C_API void bspline_score_c_mse (Bspline_optimize_data *bod);
C_API void bspline_score_g_mse (Bspline_optimize_data *bod);
C_API void bspline_score_h_mse (Bspline_optimize_data *bod);
C_API void bspline_score_i_mse (Bspline_optimize_data *bod);

#endif
