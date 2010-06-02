/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_parms_h_
#define _warp_parms_h_

#include "plm_config.h"
#include <string.h>
#include "plm_file_format.h"
#include "plm_image_type.h"
#include "plm_image_patient_position.h"
#include "plm_path.h"
#include "xio_io.h"

class Warp_parms {
public:
    /* Input files */
    char input_fn[_MAX_PATH];
    char input_ss_img[_MAX_PATH];
    char input_ss_list[_MAX_PATH];
    char input_dose_img[_MAX_PATH];
    char input_dose_xio[_MAX_PATH];
    char input_dose_ast[_MAX_PATH];
    char vf_in_fn[_MAX_PATH];
    char xf_in_fn[_MAX_PATH];
    char fixed_im_fn[_MAX_PATH];
    char ctatts_in_fn[_MAX_PATH];
    char dif_in_fn[_MAX_PATH];
    char dicom_dir[_MAX_PATH];

    /* Output files */
    char output_cxt[_MAX_PATH];
    char output_dicom[_MAX_PATH];
    char output_dij[_MAX_PATH];
    char output_dose_img[_MAX_PATH];
    char output_img[_MAX_PATH];
    char output_labelmap_fn[_MAX_PATH];
    char output_prefix[_MAX_PATH];
    char output_ss_img[_MAX_PATH];
    char output_ss_list[_MAX_PATH];
    char output_vf[_MAX_PATH];
    char output_xio_dirname[_MAX_PATH];

    /* Geometry options */
    float offset[3];
    float spacing[3];
    int dims[3];

    /* Misc options */
    float default_val;
    int interp_lin;              /* trilinear (1) or nn (0) */
    Plm_image_type output_type;
    Xio_version output_xio_version;
    int prune_empty;             /* remove empty structures (1) or not (0) */
    int use_itk;                 /* force use of itk (1) or not (0) */
    Plm_image_patient_position patient_pos;

public:
    Warp_parms () {
	memset (this, 0, sizeof(Warp_parms));
	default_val = 0.0f;
	interp_lin = 1;
	output_type = PLM_IMG_TYPE_UNDEFINED;
	output_xio_version = XIO_VERSION_4_2_1;
	prune_empty = 0;
	use_itk = 0;
	patient_pos = PATIENT_POSITION_UNKNOWN;
    }
};

#endif
