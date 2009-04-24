/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _readmha_h_
#define _readmha_h_

#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

Volume* read_mha (char* filename);
void write_mha (char* filename, Volume* vol);

#if defined __cplusplus
}
#endif

#endif
