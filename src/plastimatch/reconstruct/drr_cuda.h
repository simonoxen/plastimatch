/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_cuda_h_
#define _drr_cuda_h_

#include "plm_config.h"
#include "plmbase.h"
#include "drr_opts.h"
#include "proj_image.h"
#include "delayload.h"

#if defined __cplusplus
extern "C" {
#endif

plmcuda_EXPORT (
void* drr_cuda_state_create,
    Proj_image *proj,
    Volume *vol,
    Drr_options *options
);

plmcuda_EXPORT (
void drr_cuda_state_destroy,
    void *void_state
);

void*
drr_cuda_state_create_cu (
    Proj_image *proj,
    Volume *vol,
    Drr_options *options
);
void
drr_cuda_state_destroy_cu (
    void *void_state
);

plmcuda_EXPORT (
void drr_cuda_ray_trace_image,
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
