/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_image_h_
#define _ss_image_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_image.h"
#include "referenced_dicom_dir.h"
#include "rtss_polyline_set.h"
#include "warp_parms.h"
#include "xform.h"
#include "xio_ct.h"

class Ss_image {
public:
    Rtss_polyline_set *m_ss_list; /* Names of structures */
    Rtss_polyline_set *m_cxt;     /* Structure set in polyline form */
    Plm_image *m_ss_img;          /* Structure set in lossless bitmap form */
    Plm_image *m_labelmap;        /* Structure set lossy bitmap form */

public:
    Ss_image () {
	m_ss_list = 0;
	m_cxt = 0;
	m_ss_img = 0;
	m_labelmap = 0;
    }
    ~Ss_image () {
	if (this->m_ss_list) {
	    delete this->m_ss_list;
	}
	if (this->m_cxt) {
	    delete this->m_cxt;
	}
	if (this->m_ss_img) {
	    delete this->m_ss_img;
	}
	if (this->m_labelmap) {
	    delete this->m_labelmap;
	}
    }
    void
    clear () {
	if (this->m_ss_list) {
	    delete this->m_ss_list;
	    this->m_ss_list = 0;
	}
	if (this->m_cxt) {
	    delete this->m_cxt;
	    this->m_cxt = 0;
	}
	if (this->m_ss_img) {
	    delete this->m_ss_img;
	    this->m_ss_img = 0;
	}
	if (this->m_labelmap) {
	    delete this->m_labelmap;
	    this->m_labelmap = 0;
	}
    }
    void
    load (const char *ss_img, const char *ss_list);
    void
    load_cxt (const CBString &input_fn);
    void
    load_xio (char *input_dir);
    void
    load_gdcm_rtss (const char *input_fn, const char *dicom_dir);

    void
    save_colormap (const CBString &colormap_fn);
    void
    save_cxt (const CBString &cxt_fn, bool prune_empty);
    void
    save_gdcm_rtss (const char *output_dir, bool reload);
    void
    save_ss_image (const CBString &ss_img_fn);
    void
    save_labelmap (const CBString &labelmap_fn);
    void
    save_prefix (const CBString &output_prefix);
    void
    save_ss_list (const CBString &ss_list_fn);
    void
    save_xio (Xio_ct_transform *xio_transform, Xio_version xio_version, 
	const CBString &output_dir);
    plastimatch1_EXPORT
    UInt32ImageType::Pointer
    get_ss_img (void);
    plastimatch1_EXPORT
    Rtss_polyline_set *
    get_ss_list (void);

    void
    apply_dicom_dir (const Referenced_dicom_dir *rdd);
    void
    convert_ss_img_to_cxt (void);
    void
    cxt_re_extract (void);
    void
    prune_empty (void);
    void
    rasterize (void);
    void
    set_geometry_from_plm_image_header (Plm_image_header *pih);
    void
    warp (Xform *xf, Plm_image_header *pih, Warp_parms *parms);
};

#endif
