/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _synthetic_mha_main_h_
#define _synthetic_mha_main_h_

#include <stdlib.h>
#include "itk_image.h"
#include "plm_path.h"
#include "synthetic_mha.h"

class Synthetic_mha_main_parms {
public:
    char output_fn[_MAX_PATH];
    int have_offset;
    float volume_size[3];
    Synthetic_mha_parms sm_parms;
public:
    Synthetic_mha_main_parms () {
	*output_fn = 0;
	have_offset = 0;
	for (int i = 0; i < 3; i++) {
	    volume_size[i] = 500.0f;
	}
    }
};

#endif
