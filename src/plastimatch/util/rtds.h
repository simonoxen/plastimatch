/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plmutil_config.h"
#include <vector>
#include "plm_path.h"

#include "metadata.h"
#include "slice_index.h"

// TODO: [1] Change type of m_rdd to Slice_index*
//       [2] Change type of m_meta to Metadata*

class Gdcm_series;
//class Metadata;
class Plm_image;
class Rtss;
//class Slice_index;
class Xio_ct_transform;

/* rtds = RT data set */
class PLMUTIL_API Rtds {
public:
    Plm_image *m_img;                  /* CT image */
    Rtss *m_rtss;                      /* RT structure set */
    Plm_image *m_dose;                 /* RT dose */

    Gdcm_series *m_gdcm_series;        /* Input dicom parse info */
    Slice_index m_rdd;                 /* UIDs, etc */
    Metadata m_meta;                   /* Patient name, patient id, etc. */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
                                          coordinates */
    char m_xio_dose_input[_MAX_PATH];  /* Input XiO dose file to use as 
                                          template for XiO dose saving. */
public:
    Rtds ();
    ~Rtds ();
    void load_dicom_dir (const char *dicom_dir);
    void load_xio (const char *xio_dir, Slice_index *rdd);
    void load_ss_img (const char *ss_img, const char *ss_list);
    void load_dose_img (const char *dose_img);
    void load_dose_xio (const char *dose_xio);
    void load_dose_astroid (const char *dose_astroid);
    void load_dose_mc (const char *dose_mc);
    void load_rdd (const char *rdd);
    void load_dicom (const char *dicom_dir); 
    void load_dcmtk (const char *dicom_dir); 
    void load_gdcm (const char *dicom_dir); 
    void save_dicom (const char *output_dir);
    void save_dcmtk (const char *dicom_dir);
    void save_gdcm (const char *dicom_dir); 
    void convert_ss_img_to_cxt (void);
    void set_user_metadata (std::vector<std::string>& metadata);
    void set_dose (Plm_image *pli);
};

#endif
