/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_mask_h_
#define _pcmd_mask_h_

#include "plmcli_config.h"
#include <string.h>
#include <stdlib.h>
#include "itk_mask.h"
#include "plm_image_type.h"
#include "pstring.h"

class Mask_parms {
public:
    Pstring input_fn;
    Pstring output_fn;
    Pstring mask_fn;
    enum Mask_operation mask_operation;
    float mask_value;
    bool output_dicom;
    Plm_image_type output_type;
public:
    Mask_parms () {
	mask_operation = MASK_OPERATION_FILL;
	mask_value = 0.;
	output_dicom = false;
	output_type = PLM_IMG_TYPE_UNDEFINED;
    }
};

void do_command_mask (int argc, char *argv[]);

#endif
