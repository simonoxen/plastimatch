/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _adjust_mha_h_
#define _adjust_mha_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "itk_image.h"

class Adjust_Mha_Parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    float upper_trunc;
    bool have_upper_trunc;
    float stretch[2];
    bool have_stretch;
public:
    Adjust_Mha_Parms () {
	memset (this, 0, sizeof(Adjust_Mha_Parms));
    }
};

#endif
