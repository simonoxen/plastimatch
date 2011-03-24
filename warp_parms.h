/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_parms_h_
#define _warp_parms_h_

#include "plm_config.h"
#include <string.h>
#include "bstrwrap.h"

#include "plm_file_format.h"
#include "plm_image_type.h"
#include "plm_image_patient_position.h"
#include "xio_io.h"

class Warp_parms {
public:
    /* Input files */
    CBString input_fn;
    CBString vf_in_fn;
    CBString xf_in_fn;
    CBString referenced_dicom_dir;
    CBString input_cxt_fn;
    CBString input_ss_img_fn;
    CBString input_ss_list_fn;
    CBString input_dose_img_fn;
    CBString input_dose_xio_fn;
    CBString input_dose_ast_fn;
    CBString input_dose_mc_fn;
    CBString fixed_img_fn;

    /* Dij input files */
    CBString ctatts_in_fn;
    CBString dif_in_fn;

    /* Output files */
    CBString output_colormap_fn;
    CBString output_cxt_fn;
    CBString output_dicom;
    CBString output_dij_fn;
    CBString output_dose_img_fn;
    CBString output_img_fn;
    CBString output_labelmap_fn;
    CBString output_pointset_fn;
    CBString output_prefix;
    CBString output_ss_img_fn;
    CBString output_ss_list_fn;
    CBString output_vf_fn;
    CBString output_xio_dirname;

    /* Output options */
    Plm_image_type output_type;
    Xio_version output_xio_version;

    /* Algorithm options */
    float default_val;
    int interp_lin;              /* trilinear (1) or nn (0) */
    int prune_empty;             /* remove empty structures (1) or not (0) */
    int use_itk;                 /* force use of itk (1) or not (0) */
    int simplify_perc;	 /* percentage of points to be purged */

    /* Geometry options */
    bool m_have_dim;
    bool m_have_origin;
    bool m_have_spacing;
    int m_dim[3];
    float m_origin[3];
    float m_spacing[3];

    /* Metadata options */
    Plm_image_patient_position patient_pos;

public:
    Warp_parms () {

	/* Geometry options */
	m_have_dim = 0;
	m_have_origin = 0;
	m_have_spacing = 0;

	/* Misc options */
	default_val = 0.0f;
	interp_lin = 1;
	output_type = PLM_IMG_TYPE_UNDEFINED;
	output_xio_version = XIO_VERSION_4_2_1;
	prune_empty = 0;
	use_itk = 0;
	simplify_perc = 0;
	patient_pos = PATIENT_POSITION_UNKNOWN;
    }
};

#endif
