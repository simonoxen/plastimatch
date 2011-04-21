/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ray_trace_exact_h_
#define _ray_trace_exact_h_

#include "plm_config.h"
#include "volume.h"
#include "volume_limit.h"

typedef void (*Ray_trace_callback) (
    void *callback_data, 
    int vox_index, 
    double vox_len, 
    float vox_value);

#if defined __cplusplus
extern "C" {
#endif

void
ray_trace_exact (
    Volume *vol,                  /* Input: volume */
    Volume_limit *vol_limit,      /* Input: min/max coordinates of volume */
    Ray_trace_callback callback,  /* Input: callback function */
    void *callback_data,          /* Input: callback function private data */
    double *p1in,                 /* Input: start point for ray */
    double *p2in                  /* Input: end point for ray */
);

#if defined __cplusplus
}
#endif

#endif
