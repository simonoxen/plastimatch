/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_stats_h_
#define _vf_stats_h_

#include "plmbase_config.h"

class Volume;

PLMBASE_C_API void vf_analyze (Volume* vol);
PLMBASE_C_API void vf_analyze_strain (Volume* vol);
PLMBASE_C_API void vf_analyze_jacobian (Volume* vol);
PLMBASE_C_API void vf_analyze_second_deriv (Volume* vol);
PLMBASE_C_API void vf_analyze_mask (Volume* vol, Volume* mask);
PLMBASE_C_API void vf_analyze_strain_mask (Volume* vol, Volume* mask);
PLMBASE_C_API void vf_print_stats (Volume* vol);


#endif
