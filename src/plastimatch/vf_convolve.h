/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_convolve_h_
#define _vf_convolve_h_

#include "plm_config.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void vf_convolve_x (Volume* vf_out, Volume* vf_in, float* ker, int width);
gpuit_EXPORT
void vf_convolve_y (Volume* vf_out, Volume* vf_in, float* ker, int width);
gpuit_EXPORT
void vf_convolve_z (Volume* vf_out, Volume* vf_in, float* ker, int width);

#if defined __cplusplus
}
#endif

#endif
