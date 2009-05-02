/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_mha_main_h_
#define _warp_mha_main_h_

#include <stdlib.h>
#include "plm_path.h"
#include "itk_image.h"

class Warp_Parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    char vf_in_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    char fixed_im_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    float default_val;
    int interp_lin;
    float offset[3];
    float spacing[3];
    int dims[3];
public:
    Warp_Parms () {
	memset (this, 0, sizeof(Warp_Parms));
	interp_lin = 1;
    }
};

#endif
