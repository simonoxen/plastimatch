/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _advantech_h_
#define _advantech_h_

#include "ise_config.h"
#include "ise_error.h"
#include "ise_structs.h"

#if (ADVANTECH_FOUND)

#if defined __cplusplus
extern "C" {
#endif

Ise_Error bitflow_init (BitflowInfo* bf, unsigned int mode);

Ise_Error bitflow_open (BitflowInfo* bf, unsigned int idx, 
			unsigned int board_no, unsigned int mode,
			unsigned long fps);

Ise_Error bitflow_grab_setup (BitflowInfo *bf, int idx);
void bitflow_grab_image (unsigned short* img, BitflowInfo *bf, int idx);

#if defined __cplusplus
}
#endif

#endif
#endif
