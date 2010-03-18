/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _crop_main_h_
#define _crop_main_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Crop_Parms {
public:
    char img_in_fn[_MAX_PATH];
    char img_out_fn[_MAX_PATH];
    int crop_vox[6];
public:
    Crop_Parms () {
	memset (this, 0, sizeof(Crop_Parms));
    }
};

void do_command_crop (int argc, char *argv[]);

#endif
