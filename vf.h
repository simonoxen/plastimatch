/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_h_
#define _vf_h_

#include "plm_config.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
Volume* vf_warp (Volume* vout, Volume* vin, Volume* vf);

#if defined __cplusplus
}
#endif

#endif
