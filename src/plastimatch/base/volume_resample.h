/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_resample_h_
#define _volume_resample_h_

#include "plmbase_config.h"

PLMBASE_API Volume::Pointer volume_resample (
        Volume* vol_in,
        const plm_long* dim,
        const float* offset,
        const float* spacing
);
PLMBASE_API Volume::Pointer volume_resample (
    Volume* vol_in, const Volume_header *vh);
PLMBASE_API Volume::Pointer volume_resample_nn (
        Volume* vol_in,
        const plm_long* dim,
        const float* offset,
        const float* spacing
);
PLMBASE_API Volume::Pointer volume_subsample_vox (
    Volume* vol_in, float* sampling_rate);
PLMBASE_API Volume::Pointer volume_subsample_vox_nn (
    Volume* vol_in, float* sampling_rate);
PLMBASE_API Volume::Pointer volume_subsample_vox_legacy (
    Volume* vol_in, float* sampling_rate);
PLMBASE_API Volume::Pointer volume_subsample_vox_legacy_nn (
    Volume* vol_in, float* sampling_rate);

#endif
