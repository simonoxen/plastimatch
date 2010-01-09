/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_parms_h_
#define _warp_parms_h_

#include "plm_config.h"
#include <string.h>
#include "plm_file_format.h"
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
    char prefix[_MAX_PATH];
    char labelmap_fn[_MAX_PATH];
    char ss_img_fn[_MAX_PATH];
    char ss_list_fn[_MAX_PATH];
    float default_val;
    int use_itk;                 /* force use of itk (1) or not (0) */
    int interp_lin;              /* trilinear (1) or nn (0) */
    Plm_file_format output_format;
    PlmImageType output_type;
    float offset[3];
    float spacing[3];
    int dims[3];

public:
    Warp_parms () {
	memset (this, 0, sizeof(Warp_parms));
	use_itk = 0;
	interp_lin = 1;
	output_format = PLM_FILE_FMT_UNKNOWN;
	output_type = PLM_IMG_TYPE_UNDEFINED;
    }
};

#endif
