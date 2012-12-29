/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_h_
#define _rtss_h_

#include "plmutil_config.h"

#include "itk_image_type.h"
#include "metadata.h"
#include "pstring.h"
#include "xio_studyset.h"  /* enum Xio_version */

class Plm_image;
class Plm_image_header;
class Rtds;
class Rtss_structure_set;
class Rtss_structure;
class Slice_index;
class Xform;
class Xio_ct_transform;
class Warp_parms;

class PLMUTIL_API Rtss {
public:
    Rtss_structure_set *m_cxt;  /* Structure set in polyline form */
    Plm_image *m_ss_img;        /* Structure set in lossless bitmap form */
    Plm_image *m_labelmap;      /* Structure set lossy bitmap form */
    Metadata m_meta;            /* Metadata specific to this ss_image */

public:
    Rtss (Rtds *rtds = 0);
    ~Rtss ();

    void clear ();
    void load (const char *ss_img, const char *ss_list);
    void load_cxt (const Pstring &input_fn, Slice_index *rdd);
    void load_prefix (const char *prefix_dir);
    void load_prefix (const Pstring &prefix_dir);
    void load_xio (const Xio_studyset& xio_studyset);
    void load_gdcm_rtss (const char *input_fn,  Slice_index *rdd);

    size_t get_num_structures ();
    std::string get_structure_name (size_t index);
    UCharImageType::Pointer get_structure_image (int index);

    void save_colormap (const Pstring &colormap_fn);
    void save_cxt (Slice_index *rdd, const Pstring &cxt_fn, bool prune_empty);
    void save_gdcm_rtss (const char *output_dir, Slice_index *rdd);
    void save_fcsv (const Rtss_structure *curr_structure, const Pstring& fn);
    void save_prefix_fcsv (const Pstring &output_prefix);
    void save_ss_image (const Pstring &ss_img_fn);
    void save_labelmap (const Pstring &labelmap_fn);
    void save_prefix (const char *output_prefix);
    void save_prefix (const Pstring &output_prefix);
    void save_ss_list (const Pstring &ss_list_fn);
    void save_xio (Xio_ct_transform *xio_transform, Xio_version xio_version, 
        const Pstring &output_dir);
    UInt32ImageType::Pointer get_ss_img_uint32 (void);
    UCharVecImageType::Pointer get_ss_img_uchar_vec (void);

    void apply_dicom_dir (const Slice_index *rdd);
    void convert_ss_img_to_cxt (void);
    void convert_to_uchar_vec (void);
    void cxt_re_extract (void);
    void prune_empty (void);
    void rasterize (Plm_image_header *pih, bool want_labelmap, 
        bool xor_overlapping);
    void set_geometry (const Plm_image_header *pih);
    void find_rasterization_geometry (Plm_image_header *pih);
    void warp (Xform *xf, Plm_image_header *pih, bool use_itk = false);
    void warp (Xform *xf, Plm_image_header *pih, Warp_parms *parms);
};

#endif
