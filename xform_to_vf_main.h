/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xform_to_vf_main_h_
#define _xform_to_vf_main_h_

#include <stdlib.h>
#include "itk_image.h"

class Xform_To_Vf_Parms {
public:
    char xf_in_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    int dim[3];
    float spacing[3];
    float offset[3];
public:
    Xform_To_Vf_Parms () {
	memset (this, 0, sizeof(Xform_To_Vf_Parms));
    }
};

#endif
