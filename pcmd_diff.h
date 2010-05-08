/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_diff_h_
#define _pcmd_diff_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Diff_parms {
public:
    char img_in_1_fn[_MAX_PATH];
    char img_in_2_fn[_MAX_PATH];
    char img_out_fn[_MAX_PATH];
public:
    Diff_parms () {
	memset (this, 0, sizeof(Diff_parms));
    }
};

void do_command_diff (int argc, char *argv[]);

#endif
