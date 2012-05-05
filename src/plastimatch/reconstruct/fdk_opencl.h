/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _FDK_OPENCL_H_
#define _FDK_OPENCL_H_

#include "plmreconstruct_config.h"
#include "plmbase.h"
#include "fdk.h"
#include "proj_image_dir.h"
#include "delayload.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void opencl_reconstruct_conebeam (
    Volume *vol, 
    Proj_image_dir *proj_dir, 
    Fdk_parms *parms
);

#if defined __cplusplus
}
#endif

#endif
