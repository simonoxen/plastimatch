/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_stats_h_
#define _pcmd_stats_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "bstrwrap.h"
#include "itk_image.h"
#include "plm_path.h"

class Stats_parms {
public:
    CBString img_in_fn;
    CBString mask_fn;
};

void do_command_stats (int argc, char *argv[]);

#endif
