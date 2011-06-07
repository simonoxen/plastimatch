/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _reg_opts_h_
#define _reg_opts_h_

#include "plm_config.h"
#include "reg.h"

class Reg_options
{
public:
    char* input_vf_fn;
    Reg_parms parms;
public:
    Reg_options () {
        /* Init */
        input_vf_fn = 0;
    }
};

void
reg_opts_parse_args (Reg_options* options, int argc, char* argv[]);

#endif /* _reg_opts_h_ */
