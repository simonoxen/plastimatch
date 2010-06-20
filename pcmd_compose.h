/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_compose_h_
#define _pcmd_compose_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "itk_image.h"
#include "plm_path.h"

class Compose_parms {
public:
    char input_1[_MAX_PATH];
    char input_2[_MAX_PATH];
    char output_fn[_MAX_PATH];
    int negate_mask;
    float mask_value;
    int output_dicom;
    Plm_image_type output_type;
public:
    Compose_parms () {
	memset (this, 0, sizeof(Compose_parms));
    }
};

void do_command_compose (int argc, char *argv[]);

#endif
