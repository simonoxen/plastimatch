/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_main_h_
#define _warp_main_h_

#include "plm_config.h"
#include <string.h>
#include "plm_image_type.h"
#include "plm_path.h"

class Warp_parms {
public:
    char input_fn[_MAX_PATH];
    char output_fn[_MAX_PATH];
    char vf_in_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    char fixed_im_fn[_MAX_PATH];
    char vf_out_fn[_MAX_PATH];
    char ctatts_in_fn[_MAX_PATH];
    char dif_in_fn[_MAX_PATH];
    char dicom_dir[_MAX_PATH];
    float default_val;
    int use_itk;                 /* force use of itk (1) or not (0) */
    int interp_lin;              /* trilinear (1) or nn (0) */
    int output_dicom;
    PlmImageType output_type;
    float offset[3];
    float spacing[3];
    int dims[3];

public:
    Warp_parms () {
	memset (this, 0, sizeof(Warp_parms));
	use_itk = 0;
	interp_lin = 1;
	output_dicom = 0;
	output_type = PLM_IMG_TYPE_UNDEFINED;
    }
};

void
do_command_warp (int argc, char* argv[]);
void
warp_image_main (Warp_parms* parms);
void
warp_dij_main (Warp_parms* parms);
void
warp_pointset_main (Warp_parms* parms);
void
warp_dicom_rtss (Warp_parms* parms);

#endif
