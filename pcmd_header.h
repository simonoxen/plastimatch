/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_header_h_
#define _pcmd_header_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"

class Header_parms {
public:
    char mha_in_fn[_MAX_PATH];
public:
    Header_parms () {
	memset (this, 0, sizeof(Header_parms));
    }
};

void
do_command_header (int argc, char *argv[]);

#endif
