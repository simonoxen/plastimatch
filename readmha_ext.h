/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _readmha_h_
#define _readmha_h_

#include "volume.h"

Volume* read_mha (char* filename);
void write_mha (char* filename, Volume* vol);
void write_mha (char* filename, Volume* vol, MGHCBCT_Options_ext* options);

#endif
