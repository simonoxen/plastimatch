/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _readmha_h_ext
#define _readmha_h_ext

#include "volume.h"

Volume* read_mha_512prefix (char* filename);
void write_mha (char* filename, Volume* vol);
void write_mha_512prefix (char* filename, Volume* vol, MGHCBCT_Options_ext* options);

#endif
