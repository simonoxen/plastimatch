/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _resample_main_h_
#define _resample_main_h_

#include "plm_config.h"
#include <stdlib.h>
#include "plm_path.h"
#include "plm_image_type.h"

class Resample_parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    char input_fixed[_MAX_PATH];
    Plm_image_type output_type;
    float origin[3];
    int have_origin;
    float spacing[3];
    int have_spacing;
    int size[3];
    int have_size;
    int subsample[3];
    int have_subsample;
    float default_val;
    int have_default_val;
    int adjust;
    int interp_lin;
public:
    Resample_parms () {
	*mha_in_fn = 0;
	*mha_out_fn = 0;
	*input_fixed = 0;
	output_type = PLM_IMG_TYPE_UNDEFINED;
	for (int i = 0; i < 3; i++) {
	    origin[i] = 0.0;
	    spacing[i] = 0.0;
	    size[i] = 0;
	    subsample[i] = 0;
	}
	have_origin = 0;
	have_spacing = 0;
	have_size = 0;
	have_subsample = 0;
	default_val = 0.0;
	have_default_val = 0;
	adjust = 0;
	interp_lin=1;
    }
};

void
do_command_resample (int argc, char *argv[]);

#endif
