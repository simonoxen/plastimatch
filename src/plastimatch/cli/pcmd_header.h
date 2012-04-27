/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_header_h_
#define _pcmd_header_h_

#include "plmcli_config.h"
#include <string.h>
#include <stdlib.h>
#include "pstring.h"

class Header_parms {
public:
    Pstring img_in_fn;
};

void
do_command_header (int argc, char *argv[]);

#endif
