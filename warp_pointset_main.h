/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_pointset_main_h_
#define _warp_pointset_main_h_

#include <stdlib.h>
#include "itk_image.h"

class Warp_Pointset_Parms {
public:
    char ps_in_fn[_MAX_PATH];
    char ps_out_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
public:
    Warp_Pointset_Parms () {
	memset (this, 0, sizeof(Warp_Pointset_Parms));
    }
};

#endif
