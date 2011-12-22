/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_resample_h_
#define _pcmd_resample_h_

#include "plm_config.h"
#include <stdlib.h>
#include "direction_cosines.h"
#include "plm_image_type.h"
#include "pstring.h"

class Resample_parms {
public:
    Pstring input_fn;
    Pstring output_fn;
    Pstring fixed_fn;
    Plm_image_type output_type;
    size_t dim[3];
    bool m_have_dim;
    float origin[3];
    bool m_have_origin;
    float spacing[3];
    bool m_have_spacing;
    int subsample[3];
    bool m_have_subsample;
    Direction_cosines m_dc;
    bool m_have_direction_cosines;
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
	    dim[i] = 0;
	    subsample[i] = 0;
	}
	m_have_dim = false;
	m_have_origin = false;
	m_have_spacing = false;
	m_have_subsample = false;
	default_val = 0.0;
	have_default_val = false;
	adjust = 0;
	interp_lin=true;
    }
};

void
do_command_resample (int argc, char *argv[]);

#endif
