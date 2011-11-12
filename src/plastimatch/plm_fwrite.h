/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_fwrite_h_
#define _plm_fwrite_h_

#include "plm_config.h"
#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT void 
plm_fwrite (void* buf, size_t size, size_t count, FILE* fp, 
    bool force_little_endian);

#if defined __cplusplus
}
#endif

#endif
