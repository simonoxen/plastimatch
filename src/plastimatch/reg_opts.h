/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _reg_opts_h_
#define _reg_opts_h_

#include "plm_config.h"
#include "bspline_regularize.h"
#include "plm_int.h"

class Reg_options
{
public:
    char* input_vf_fn;
    char* input_xf_fn;
    Reg_parms parms;
    plm_long vox_per_rgn[3];
public:
    Reg_options () {
        /* Init */
        input_vf_fn = 0;
        input_xf_fn = 0;
    	for (int d = 0; d < 3; d++) {
    	    vox_per_rgn[d] = 15;
    	}
    }
};

void
reg_opts_parse_args (Reg_options* options, int argc, char* argv[]);

#endif /* _reg_opts_h_ */
