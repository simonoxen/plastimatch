/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_diff_h_
#define _pcmd_diff_h_

#include "plmcli_config.h"
#include <string.h>
#include <stdlib.h>
#include "itk_image.h"
#include "pstring.h"

class plastimatch1_EXPORT Diff_parms {
public:
    Pstring img_in_1_fn;
    Pstring img_in_2_fn;
    Pstring img_out_fn;
};

void do_command_diff (int argc, char *argv[]);

plastimatch1_EXPORT
void diff_main (Diff_parms* parms);

#endif
