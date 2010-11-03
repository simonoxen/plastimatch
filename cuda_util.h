/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_util_h_
#define _cuda_util_h_

#include "plm_config.h"
#include <cuda.h>

#define GRID_LIMIT_X 65535
#define GRID_LIMIT_Y 65535


#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT void
CUDA_check_error (const char *msg);

gpuit_EXPORT int
CUDA_detect_error ();

gpuit_EXPORT void
CUDA_listgpu ();

gpuit_EXPORT void
CUDA_selectgpu (int gpuid);

gpuit_EXPORT int
CUDA_getarch (int gpuid);


#if defined __cplusplus
}
#endif

#endif
