/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demons_cuda_h_
#define _demons_cuda_h_

#include "plm_config.h"

#include "demons.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif
plmcuda_EXPORT
Volume* demons_cuda (Volume* fixed, Volume* moving, Volume* moving_grad, Volume* vf_init, DEMONS_Parms* parms);
#if defined __cplusplus
}
#endif

#endif
