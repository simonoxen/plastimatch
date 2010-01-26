/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mask_main_h_
#define _mask_main_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Mask_Parms {
public:
    char input_fn[_MAX_PATH];
    char output_fn[_MAX_PATH];
    char mask_fn[_MAX_PATH];
    int negate_mask;
    float mask_value;
    int output_dicom;
    PlmImageType output_type;
public:
    Mask_Parms () {
	memset (this, 0, sizeof(Mask_Parms));
    }
};

void do_command_mask (int argc, char *argv[]);

#endif
