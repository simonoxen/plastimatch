/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _DRR_OPENCL_H_
#define _DRR_OPENCL_H_

#include "plmreconstruct_config.h"
#include "plmbase.h"
#include "drr_opts.h"
#include "proj_image.h"
#include "delayload.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void* drr_opencl_state_create (
    Proj_image *proj,
    Volume *vol,
    Drr_options *options
);

gpuit_EXPORT
void drr_opencl_state_destroy (
    void *dev_state
);

gpuit_EXPORT
void
drr_opencl_ray_trace_image (
    Proj_image *proj, 
    Volume *vol, 
    Volume_limit *vol_limit, 
    double p1[3], 
    double ul_room[3], 
    double incr_r[3], 
    double incr_c[3], 
    void *dev_state, 
    Drr_options *options
);

#if defined __cplusplus
}
#endif

#endif
