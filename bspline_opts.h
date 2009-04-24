/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_opts_h_
#define _bspline_opts_h_

#include "bspline.h"

#define OPTION_INTEGER_SPACING_STRING "integer-spacing"
#define OPTION_INTEGER_SPACING 's'

#define OPTION_RESOLUTION 'r'
#define OPTION_OUTPUT_FILE 'O'
#define OPTION_ALGORITHM 'a'
#define OPTION_MAX_ITS 'm'

typedef struct BSPLINE_Options_struct BSPLINE_Options;
struct BSPLINE_Options_struct {
    char* fixed_fn;
    char* moving_fn;
    char* output_fn;
    char* method;
    int vox_per_rgn[3];
    BSPLINE_Parms parms;
};

void parse_args (BSPLINE_Options* options, int argc, char* argv[]);
void bspline_opts_set_default_options (BSPLINE_Options* options);

#endif
