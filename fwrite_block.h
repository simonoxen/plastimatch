/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _fwrite_block_h_
#define _fwrite_block_h_

#include "plm_config.h"
#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT void 
fwrite_block (void* buf, size_t size, size_t count, FILE* fp);

#if defined __cplusplus
}
#endif

#endif
