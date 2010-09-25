#ifndef _FDK_OPENCL_H_
#define _FDK_OPENCL_H_

#include "fdk_opts.h"
#include "proj_image_dir.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

void OPENCL_reconstruct_conebeam_and_convert_to_hu (Volume *vol, Proj_image_dir *proj_dir, Fdk_options *options);

#if defined __cplusplus
}
#endif

#endif

