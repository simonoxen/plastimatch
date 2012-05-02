/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_resample_h_
#define _volume_resample_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

API Volume* volume_resample (
        Volume* vol_in,
        const plm_long* dim,
        const float* offset,
        const float* spacing
);
API Volume* volume_resample (Volume* vol_in, const Volume_header *vh);
C_API Volume* volume_resample_nn (
        Volume* vol_in,
        const plm_long* dim,
        const float* offset,
        const float* spacing
);
C_API Volume* volume_subsample (Volume* vol_in, int* sampling_rate);
C_API Volume* volume_subsample_nn (Volume* vol_in, int* sampling_rate);

#endif
