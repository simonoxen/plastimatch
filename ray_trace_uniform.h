/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ray_trace_uniform_h_
#define _ray_trace_uniform_h_

#include "plm_config.h"
#include "volume.h"
#include "volume_limit.h"

#if defined __cplusplus
extern "C" {
#endif

void
ray_trace_uniform (
    Volume *vol,                  /* Input: volume */
    Volume_limit *vol_limit,      /* Input: min/max coordinates of volume */
    Ray_trace_callback callback,  /* Input: callback function */
    void *callback_data,          /* Input: callback function private data */
    double *p1in,                 /* Input: start point for ray */
    double *p2in,                 /* Input: end point for ray */
    float ray_step                /* Input: uniform step size */
);

#if defined __cplusplus
}
#endif

#endif
