/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_h_
#define _rpl_volume_h_

#include "plm_config.h"
#include "volume.h"

typedef struct rpl_volume Rpl_volume;
struct rpl_volume {
    Volume *vol;
    double *ap_ul;
    double *incr_r;
    double *incr_c;
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT 
Rpl_volume*
rpl_volume_create (
    Volume* ct_vol, // ct volume
    int ires[2],    // aperture dimensions
    float ray_step  // uniform ray step size
);

#if defined __cplusplus
}
#endif

#endif
