/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_stats_h_
#define _vf_stats_h_

#include "plm_config.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT 
void
vf_analyze (Volume* vol);
gpuit_EXPORT 
void
vf_analyze_strain (Volume* vol);

void
vf_analyze_mask (Volume* vol, Volume* mask);

void
vf_analyze_strain_mask (Volume* vol, Volume* mask);

#if defined __cplusplus
}
#endif

#endif
