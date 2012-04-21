/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_resample_h_
#define _volume_resample_h_

#include "plm_config.h"

class Volume;
class Volume_header;

gpuit_EXPORT
Volume* volume_resample (Volume* vol_in, const plm_long* dim, const float* offset, const float* spacing);
gpuit_EXPORT
Volume* volume_resample_nn (Volume* vol_in, const plm_long* dim, const float* offset, const float* spacing);
gpuit_EXPORT
Volume* volume_resample (Volume* vol_in, const Volume_header *vh);

gpuit_EXPORT
Volume* volume_subsample (Volume* vol_in, int* sampling_rate);
gpuit_EXPORT
Volume* volume_subsample_nn (Volume* vol_in, int* sampling_rate);

#endif
