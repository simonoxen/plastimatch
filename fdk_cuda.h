/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_cuda_h_
#define _fdk_cuda_h_

#include "fdk_opts.h"
#include "proj_image_dir.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

int 
CUDA_reconstruct_conebeam (
    Volume *vol, 
    Proj_image_dir *proj_dir,
    Fdk_options *options
);

#if defined __cplusplus
}
#endif

#endif
