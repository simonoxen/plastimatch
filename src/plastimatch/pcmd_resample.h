/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_resample_h_
#define _pcmd_resample_h_

#include "plm_config.h"
#include <stdlib.h>
#include "bstrwrap.h"
#include "plm_image_type.h"

class Resample_parms {
public:
    CBString img_in_fn;
    CBString img_out_fn;
    CBString fixed_fn;
    Plm_image_type output_type;
    float origin[3];
    bool have_origin;
    float spacing[3];
    bool have_spacing;
    int size[3];
    bool have_size;
    int subsample[3];
    bool have_subsample;
    float default_val;
    bool have_default_val;
    int adjust;
    bool interp_lin;
public:
    Resample_parms () {
	output_type = PLM_IMG_TYPE_UNDEFINED;
	for (int i = 0; i < 3; i++) {
	    origin[i] = 0.0;
	    spacing[i] = 0.0;
	    size[i] = 0;
	    subsample[i] = 0;
	}
	have_origin = false;
	have_spacing = false;
	have_size = false;
	have_subsample = false;
	default_val = 0.0;
	have_default_val = false;
	adjust = 0;
	interp_lin=true;
    }
};

void
do_command_resample (int argc, char *argv[]);

#endif
