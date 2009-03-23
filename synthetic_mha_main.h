/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _synthetic_mha_main_h_
#define _synthetic_mha_main_h_

#include <stdlib.h>
#include "itk_image.h"

enum Pattern_type {
    PATTERN_GAUSS,
    PATTERN_RECT
};

class Synthetic_mha_parms {
public:
    char output_fn[_MAX_PATH];
    int output_type;
    Pattern_type pattern;
    float origin[3];
    int have_origin;
    float volume_size[3];
    int res[3];
    float background;
    float foreground;
    float gauss_center[3];
    float gauss_std[3];
public:
    Synthetic_mha_parms () {
	*output_fn = 0;
	output_type = PLM_IMG_TYPE_UNDEFINED;
	pattern = PATTERN_GAUSS;
	for (int i = 0; i < 3; i++) {
	    origin[i] = 0.0f;
	    volume_size[i] = 500.0f;
	    res[i] = 100;
	    gauss_center[i] = 0.0f;
	    gauss_std[i] = 100.0f;
	}
	have_origin = 0;
	background = -1000.0f;
	foreground = 0.0f;
    }
};

#endif
