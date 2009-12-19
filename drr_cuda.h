/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_cuda_h_
#define _drr_cuda_h_

#include "plm_config.h"
#include "drr_opts.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

int CUDA_DRR3 (Volume *vol, Drr_options *options);
int CUDA_DRR (Volume *vol, Drr_options *options);

void
drr_cuda_render_volume_perspective (
    Proj_image *proj,
    Volume *vol, 
    double ps[2], 
    char *multispectral_fn, 
    Drr_options *options
);

#if defined __cplusplus
}
#endif

#endif
