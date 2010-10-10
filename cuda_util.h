/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_utils_h_
#define _cuda_utils_h_

#include "plm_config.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
cuda_utils_check_error (const char *msg);

#if defined __cplusplus
}
#endif

#endif
