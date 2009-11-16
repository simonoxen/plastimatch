/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _adjust_main_h_
#define _adjust_main_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Adjust_Parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    float truncate_above;
    bool have_truncate_above;
    float truncate_below;
    bool have_truncate_below;
    float stretch[2];
    bool have_stretch;
    int output_dicom;
    PlmImageType output_type;
public:
    Adjust_Parms () {
	memset (this, 0, sizeof(Adjust_Parms));
    }
};

void do_command_adjust (int argc, char *argv[]);

#endif
