/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_slice_h_
#define _pcmd_slice_h_

#include "plm_config.h"
#include <string.h>
#include <stdlib.h>
#include "bstrwrap.h"

class Slice_parms {
public:
    CBString img_in_fn;
    CBString img_out_fn;
    int thumbnail_dim;
    float thumbnail_spacing;
    bool have_slice_loc;
    float slice_loc;
public:
    Slice_parms () {
	img_out_fn = "thumb.mhd";
	thumbnail_dim = 16;
	thumbnail_spacing = 30.0;
	slice_loc = 0.0;
	have_slice_loc = false;
    }
};

void
do_command_slice (int argc, char *argv[]);

#endif
