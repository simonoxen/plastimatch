/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_mha_main_h_
#define _warp_mha_main_h_

#include <stdlib.h>
#include "itk_image.h"


class Warp_Parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    char vf_in_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    char fixed_im_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    int output_type;
    float default_val;
public:
    Warp_Parms () {
	*mha_in_fn = 0;
	*mha_out_fn = 0;
	*vf_in_fn = 0;
	*xf_in_fn = 0;
	*fixed_im_fn = 0;
	*vf_out_fn = 0;
	output_type = TYPE_UNSPECIFIED;
	default_val = 0.0;
    }
};

#endif
