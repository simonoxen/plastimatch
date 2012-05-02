/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_stats_h_
#define _vf_stats_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

class Volume;

C_API void vf_analyze (Volume* vol);
C_API void vf_analyze_strain (Volume* vol);
C_API void vf_analyze_jacobian (Volume* vol);
C_API void vf_analyze_second_deriv (Volume* vol);
C_API void vf_analyze_mask (Volume* vol, Volume* mask);
C_API void vf_analyze_strain_mask (Volume* vol, Volume* mask);
C_API void vf_print_stats (Volume* vol);


#endif
