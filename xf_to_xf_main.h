/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xf_to_xf_main_h_
#define _xf_to_xf_main_h_

#include <stdlib.h>
#include "xform.h"

class Xf_To_Xf_Parms {
public:
    char xf_in_fn[_MAX_PATH];
    char xf_out_fn[_MAX_PATH];
    XFormInternalType xf_type;
    int dim[3];
    float spacing[3];
    float offset[3];
public:
    Xf_To_Xf_Parms () {
	memset (this, 0, sizeof(Xf_To_Xf_Parms));
    }
};

#endif
