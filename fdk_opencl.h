/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _FDK_OPENCL_H_
#define _FDK_OPENCL_H_

#include "plm_config.h"
#include "fdk_opts.h"
#include "proj_image_dir.h"
#include "volume.h"
#include "delayload.h"

#if defined __cplusplus
extern "C" {
#endif

plmopencl_EXPORT (
void OPENCL_reconstruct_conebeam_and_convert_to_hu,
    Volume *vol,
    Proj_image_dir *proj_dir,
    Fdk_options *options
);

plmopencl_EXPORT (
void opencl_reconstruct_conebeam,
    Volume *vol, 
    Proj_image_dir *proj_dir, 
    Fdk_options *options
);

#if defined __cplusplus
}
#endif

#endif
