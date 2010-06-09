/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_stats_h_
#define _pcmd_stats_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Stats_parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mask_fn[_MAX_PATH];
public:
    Stats_parms () {
	memset (this, 0, sizeof(Stats_parms));
    }
};

void do_command_stats (int argc, char *argv[]);

#endif
