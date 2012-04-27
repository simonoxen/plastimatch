/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _pcmd_thumbnail_h_
#define _pcmd_thumbnail_h_

#include "plm_config.h"
#include <stdlib.h>
#include <string.h>
#include "pstring.h"

class Thumbnail_parms {
public:
    Pstring img_in_fn;
    Pstring img_out_fn;
    int thumbnail_dim;
    float thumbnail_spacing;
    bool have_slice_loc;
    float slice_loc;
public:
    Thumbnail_parms () {
	img_out_fn = "thumb.mhd";
	thumbnail_dim = 16;
	thumbnail_spacing = 30.0;
	slice_loc = 0.0;
	have_slice_loc = false;
    }
};

void
do_command_thumbnail (int argc, char *argv[]);

#endif
