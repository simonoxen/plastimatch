/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _synthetic_vf_h_
#define _synthetic_vf_h_

#include "plm_config.h"
#include "itk_image.h"

enum Pattern_type {
    PATTERN_GAUSS,
    PATTERN_RECT,
    PATTERN_SPHERE
};

class Synthetic_vf_parms {
public:
    int output_type;
    Pattern_type pattern;
    int dim[3];
    float origin[3];
    float spacing[3];

    float background;
    float foreground;
    float gauss_center[3];
    float gauss_std[3];
    float rect_size[6];
    float sphere_center[3];
    float sphere_radius[3];
public:
    Synthetic_vf_parms () {
	output_type = PLM_IMG_TYPE_ITK_FLOAT;
	pattern = PATTERN_GAUSS;
	for (int i = 0; i < 3; i++) {
	    spacing[i] = 5.0f;
	    dim[i] = 100;
	    origin[i] = 0.0f;
	    gauss_center[i] = 0.0f;
	    gauss_std[i] = 100.0f;
	    sphere_center[i] = 0.0f;
	    sphere_radius[i] = 50.0f;
	}
	background = -1000.0f;
	foreground = 0.0f;
	rect_size[0] = -50.0f;
	rect_size[1] = +50.0f;
	rect_size[2] = -50.0f;
	rect_size[3] = +50.0f;
	rect_size[4] = -50.0f;
	rect_size[5] = +50.0f;
    }
};

plastimatch1_EXPORT FloatImageType::Pointer synthetic_vf (Synthetic_vf_parms* parms);
#endif
