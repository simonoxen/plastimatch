/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _readmha_h_ext
#define _readmha_h_ext

#include "volume.h"
#include "fdk_opts.h"

Volume* read_mha_512prefix (char* filename);
//void write_mha (char* filename, Volume* vol);
void write_mha_512prefix (char* filename, Volume* vol, Fdk_options* options);
//void write_mha_512 (char* filename, Volume* vol)

#endif
