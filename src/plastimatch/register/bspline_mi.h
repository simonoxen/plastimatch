/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mi_h_
#define _bspline_mi_h_

#include "plmregister_config.h"

class Bspline_optimize_data;
class Bspline_parms;

C_API void bspline_initialize_mi (Bspline_parms* parms);
C_API void bspline_score_c_mi (Bspline_optimize_data *bod);
C_API void bspline_score_d_mi (Bspline_optimize_data *bod);
C_API void bspline_score_e_mi (Bspline_optimize_data *bod);
C_API void bspline_score_f_mi (Bspline_optimize_data *bod);
C_API void bspline_score_g_mi (Bspline_optimize_data *bod);
C_API void bspline_score_h_mi (Bspline_optimize_data *bod);

#endif
