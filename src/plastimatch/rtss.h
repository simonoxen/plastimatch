/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_h_
#define _rtss_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "referenced_dicom_dir.h"
#include "rtss_polyline_set.h"
#include "warp_parms.h"
#include "xform.h"
#include "xio_ct.h"

class Rtds;

class plastimatch1_EXPORT Rtss {
public:
    Rtss_polyline_set *m_ss_list; /* Names of structures */
    Rtss_polyline_set *m_cxt;     /* Structure set in polyline form */
    Plm_image *m_ss_img;          /* Structure set in lossless bitmap form */
    Plm_image *m_labelmap;        /* Structure set lossy bitmap form */
    Img_metadata m_img_metadata;  /* Metadata specific to this ss_image */

public:
    Rtss (Rtds *rtds);
    ~Rtss ();

    void clear ();
    void load (const char *ss_img, const char *ss_list);
    void load_cxt (const CBString &input_fn, Referenced_dicom_dir *rdd);
    void load_xio (char *input_dir);
    void load_gdcm_rtss (const char *input_fn,  Referenced_dicom_dir *rdd);

    void save_colormap (const CBString &colormap_fn);
    void save_cxt (Referenced_dicom_dir *rdd, const CBString &cxt_fn, 
	bool prune_empty);
    void save_gdcm_rtss (const char *output_dir, Referenced_dicom_dir *rdd);
    void save_ss_image (const CBString &ss_img_fn);
    void
    save_labelmap (const CBString &labelmap_fn);
    void
    save_prefix (const CBString &output_prefix);
    void
    save_ss_list (const CBString &ss_list_fn);
    void
    save_xio (Xio_ct_transform *xio_transform, Xio_version xio_version, 
	const CBString &output_dir);
    UInt32ImageType::Pointer
    get_ss_img (void);
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
    void rasterize (Plm_image_header *pih, bool want_labelmap);
    void
    set_geometry_from_plm_image_header (Plm_image_header *pih);
    void
    find_rasterization_geometry (Plm_image_header *pih);
    void
    warp (Xform *xf, Plm_image_header *pih, Warp_parms *parms);
};

#endif
