/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_mask_h_
#define _pcmd_mask_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "bstrwrap.h"
#include "itk_image.h"

class Mask_Parms {
public:
    CBString input_fn;
    CBString output_fn;
    CBString mask_fn;
    bool negate_mask;
    float mask_value;
    bool output_dicom;
    Plm_image_type output_type;
public:
    Mask_Parms () {
	negate_mask = false;
	mask_value = 0.;
	output_dicom = false;
	output_type = PLM_IMG_TYPE_UNDEFINED;
    }
};

void do_command_mask (int argc, char *argv[]);

#endif
