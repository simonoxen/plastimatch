/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_opts_h_
#define _bspline_opts_h_

#include "bspline.h"

//#define OPTION_INTEGER_SPACING_STRING "integer-spacing"
//#define OPTION_INTEGER_SPACING 's'
//#define OPTION_RESOLUTION 'r'
//#define OPTION_OUTPUT_FILE 'O'
//#define OPTION_ALGORITHM 'a'
//#define OPTION_MAX_ITS 'm'

typedef struct BSPLINE_Options_struct BSPLINE_Options;
struct BSPLINE_Options_struct {
    char* fixed_fn;
    char* moving_fn;
    char* input_xf_fn;
    char* output_warped_fn;
    char* output_xf_fn;
    char* output_vf_fn;
    char* fixed_landmarks;
    char* moving_landmarks;
    char* warped_landmarks;
    char* method;
    float landmark_stiffness;
    float young_modulus;
    int vox_per_rgn[3];
    BSPLINE_Parms parms;
};

gpuit_EXPORT
void bspline_opts_parse_args (BSPLINE_Options* options, int argc, char* argv[]);

#endif
