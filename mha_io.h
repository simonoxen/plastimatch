/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mha_io_h_
#define _mha_io_h_

#include "plm_config.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Volume* read_mha (const char* filename);
gpuit_EXPORT void write_mha (const char* filename, Volume* vol);

#if defined __cplusplus
}
#endif

#endif
