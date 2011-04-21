/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_crop_h_
#define _pcmd_crop_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "bstrwrap.h"
#include "itk_image.h"

class Crop_Parms {
public:
    CBString img_in_fn;
    CBString img_out_fn;
    int crop_vox[6];
};

void do_command_crop (int argc, char *argv[]);

#endif
