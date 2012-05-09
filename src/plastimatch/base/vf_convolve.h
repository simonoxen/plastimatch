/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_convolve_h_
#define _vf_convolve_h_

/**
*  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmbase.h"
 */

#include "plmbase_config.h"

PLMBASE_C_API void vf_convolve_x (Volume* vf_out, Volume* vf_in, float* ker, int width);
PLMBASE_C_API void vf_convolve_y (Volume* vf_out, Volume* vf_in, float* ker, int width);
PLMBASE_C_API void vf_convolve_z (Volume* vf_out, Volume* vf_in, float* ker, int width);

#endif
