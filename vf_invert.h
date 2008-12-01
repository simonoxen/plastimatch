/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _vf_invert_h_
#define _vf_invert_h_

#include <stdlib.h>
#include "xform.h"
#include "plm_image.h"

class Vf_Invert_Parms {
public:
    char vf_in_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    float origin[3];
    float spacing[3];
    int dim[3];
    char fixed_img_fn[_MAX_PATH];
public:
    Vf_Invert_Parms () {
	memset (this, 0, sizeof(Vf_Invert_Parms));
    }
};

#endif
