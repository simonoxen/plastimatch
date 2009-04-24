/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demon_opts_h_
#define _demon_opts_h_

#include "demons.h"

typedef struct DEMONS_Options_struct DEMONS_Options;
struct DEMONS_Options_struct {
    char* fixed_fn;
    char* moving_fn;
    char* output_fn;
    char* method;
    DEMONS_Parms parms;
};

void parse_args (DEMONS_Options* options, int argc, char* argv[]);

#endif
