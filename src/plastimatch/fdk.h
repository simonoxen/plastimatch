/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fdk_h_
#define _fdk_h_

#include "plm_config.h"
#include "plmbase.h"
#include "fdk_opts.h"
#include "proj_image.h"
#include "proj_image_dir.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
reconstruct_conebeam (
    Volume* vol, 
    Proj_image_dir *proj_dir, 
    Fdk_options* options
);

gpuit_EXPORT
void
CUDA_reconstruct_conebeam (
    Volume *vol, 
    Proj_image_dir *proj_dir,
    Fdk_options *options
);

gpuit_EXPORT
void
fdk_do_bowtie (Volume* vol, Fdk_options* options);

#if defined __cplusplus
}
#endif

#endif
