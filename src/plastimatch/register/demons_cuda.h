/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demons_cuda_h_
#define _demons_cuda_h_

#include "plmregister_config.h"

#include "delayload.h"
#include "demons.h"

class Demons_state;
class Volume;

#if defined __cplusplus
extern "C" {
#endif

plmcuda_EXPORT (
void demons_cuda,
    Demons_state *demons_state,
    Volume* fixed,
    Volume* moving,
    Volume* moving_grad,
    Volume* vf_init,
    DEMONS_Parms* parms
);

#if defined __cplusplus
}
#endif

#endif
