/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_compare_h_
#define _pcmd_compare_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Compare_parms {
public:
    char img_in_1_fn[_MAX_PATH];
    char img_in_2_fn[_MAX_PATH];
public:
    Compare_parms () {
	memset (this, 0, sizeof(Compare_parms));
    }
};

void do_command_compare (int argc, char *argv[]);

#endif
