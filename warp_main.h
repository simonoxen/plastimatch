/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_main_h_
#define _warp_main_h_

#include "plm_config.h"
#include <string.h>
#include "plm_file_format.h"
#include "plm_image_type.h"
#include "plm_path.h"
#include "rtss_warp.h"
#include "warp_parms.h"

void
do_command_warp (int argc, char* argv[]);
void
warp_image_main (Warp_parms* parms);
void
warp_dij_main (Warp_parms* parms);
void
warp_pointset_main (Warp_parms* parms);

#endif
