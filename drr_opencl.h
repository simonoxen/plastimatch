/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _DRR_OPENCL_H_
#define _DRR_OPENCL_H_
#include "plm_config.h"

#include "volume.h"
#include "drr_opts.h"

#if defined __cplusplus
extern "C" {
#endif

void drr_opencl_render_volume (Volume* vol, Drr_options* options);

#if defined __cplusplus
}
#endif

#endif
