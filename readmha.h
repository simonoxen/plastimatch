/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _readmha_h_
#define _readmha_h_

#include "plm_config.h"
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Volume* read_mha (char* filename);
gpuit_EXPORT void write_mha (char* filename, Volume* vol);

#if defined __cplusplus
}
#endif

#endif
