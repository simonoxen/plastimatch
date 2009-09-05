/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_main_h_
#define _warp_main_h_

#include "plm_config.h"
#include <string.h>
#include "plm_path.h"

class Warp_Parms {
public:
    char mha_in_fn[_MAX_PATH];
    char mha_out_fn[_MAX_PATH];
    char vf_in_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    char fixed_im_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    char ctatts_in_fn[_MAX_PATH];
    char dif_in_fn[_MAX_PATH];
    float default_val;
    int interp_lin;
    int output_dicom;
    float offset[3];
    float spacing[3];
    int dims[3];

public:
    Warp_Parms () {
	memset (this, 0, sizeof(Warp_Parms));
	output_dicom = 0;
	interp_lin = 1;
    }
};

void
do_command_warp (int argc, char* argv[]);
void
warp_image_main (Warp_Parms* parms);
void
warp_dij_main (Warp_Parms* parms);
void
warp_pointset_main (Warp_Parms* parms);

#endif
