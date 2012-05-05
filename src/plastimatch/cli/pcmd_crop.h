/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_crop_h_
#define _pcmd_crop_h_

#include "plmcli_config.h"
#include <string.h>
#include <stdlib.h>
#include "pstring.h"

class Crop_Parms {
public:
    Pstring img_in_fn;
    Pstring img_out_fn;
    int crop_vox[6];
};

void do_command_crop (int argc, char *argv[]);

#endif
