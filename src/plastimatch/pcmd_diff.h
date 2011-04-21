/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_diff_h_
#define _pcmd_diff_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "bstrwrap.h"
#include "itk_image.h"

class Diff_parms {
public:
    CBString img_in_1_fn;
    CBString img_in_2_fn;
    CBString img_out_fn;
};

void do_command_diff (int argc, char *argv[]);

#endif
