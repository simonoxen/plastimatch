/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demons_h_
#define _demons_h_

#include "plm_config.h"
#include "plmbase.h"

#include "delayload.h"
#include "threading.h"

typedef struct DEMONS_Parms_struct DEMONS_Parms;
struct DEMONS_Parms_struct {
    Threading threading;
    float denominator_eps;
    float homog;
    float accel;
    int filter_width[3];
    int max_its;
    float filter_std;
};

#if defined __cplusplus
extern "C" {
#endif
gpuit_EXPORT
void demons_default_parms (DEMONS_Parms* parms);
gpuit_EXPORT
Volume* demons (Volume* fixed, Volume* moving, Volume* moving_grad, Volume* vf_init, DEMONS_Parms* parms);
Volume* demons_c (Volume* fixed, Volume* moving, Volume* moving_grad, Volume* vf_init, DEMONS_Parms* parms);

//plmopencl_EXPORT (
Volume* demons_opencl (
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
