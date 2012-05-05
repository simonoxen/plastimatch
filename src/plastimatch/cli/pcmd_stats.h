/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_stats_h_
#define _pcmd_stats_h_

#include "plmcli_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "pstring.h"

class Stats_parms {
public:
    Pstring img_in_fn;
    Pstring mask_fn;
};

void do_command_stats (int argc, char *argv[]);

#endif
