/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _warp_parms_h_
#define _warp_parms_h_

#include "plmutil_config.h"
#include <string.h>
#include "direction_cosines.h"
#include "plm_image_type.h"
#include "plm_int.h"
#include "pstring.h"
#include "xio_studyset.h"

class Warp_parms {
public:
    /* Input files */
    Pstring input_fn;
    Pstring xf_in_fn;
    Pstring referenced_dicom_dir;
    Pstring input_cxt_fn;
    Pstring input_prefix;
    Pstring input_ss_img_fn;
    Pstring input_ss_list_fn;
    Pstring input_dose_img_fn;
    Pstring input_dose_xio_fn;
    Pstring input_dose_ast_fn;
    Pstring input_dose_mc_fn;
    Pstring fixed_img_fn;

    /* Dij input files */
    Pstring ctatts_in_fn;
    Pstring dif_in_fn;

    /* Output files */
    Pstring output_colormap_fn;
    Pstring output_cxt_fn;
    Pstring output_dicom;
    Pstring output_dij_fn;
    Pstring output_dose_img_fn;
    Pstring output_img_fn;
    Pstring output_labelmap_fn;
    Pstring output_pointset_fn;
    std::string output_prefix;
    Pstring output_prefix_fcsv;
    Pstring output_ss_img_fn;
    Pstring output_ss_list_fn;
    Pstring output_vf_fn;
    Pstring output_xio_dirname;

    /* Output options */
    Plm_image_type output_type;
    std::string prefix_format;
    Xio_version output_xio_version;

    /* Algorithm options */
    float default_val;
    bool have_dose_scale;       /* should we scale the dose image? */
    float dose_scale;           /* how much to scale the dose image */
    int interp_lin;             /* trilinear (1) or nn (0) */
    int prune_empty;            /* remove empty structures (1) or not (0) */
    int use_itk;                /* force use of itk (1) or not (0) */
    int simplify_perc;          /* percentage of points to be purged */
    bool xor_contours;          /* or/xor overlapping structure contours */

    /* Geometry options */
    bool m_have_dim;
    bool m_have_origin;
    bool m_have_spacing;
    bool m_have_direction_cosines;
    plm_long m_dim[3];
    float m_origin[3];
    float m_spacing[3];
    Direction_cosines m_dc;

    /* Metadata options */
    std::vector<std::string> m_metadata;

public:
    Warp_parms () {

	/* Geometry options */
	m_have_dim = false;
	m_have_origin = false;
	m_have_spacing = false;
	m_have_direction_cosines = false;

	/* Misc options */
	have_dose_scale = false;
	dose_scale = 1.0f;
	default_val = 0.0f;
	interp_lin = 1;
	prefix_format = "mha";
	output_type = PLM_IMG_TYPE_UNDEFINED;
	output_xio_version = XIO_VERSION_4_2_1;
	prune_empty = 0;
	use_itk = 0;
	simplify_perc = 0;
	xor_contours = false;
    }
};

#endif
