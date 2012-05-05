/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plm_config.h"
#include <vector>

#include "metadata.h"
#include "plm_image.h"
#include "plm_path.h"
#include "slice_index.h"
#include "xio_ct.h"

class Gdcm_series;
class Rtss;

/* rtds = RT data set */
class plastimatch1_EXPORT Rtds {
public:
    Plm_image *m_img;                  /* CT image */
    Rtss *m_rtss;                      /* RT structure set */
    Plm_image *m_dose;                 /* RT dose */

    Gdcm_series *m_gdcm_series;        /* Input dicom parse info */
    Slice_index m_rdd;        /* UIDs, etc */
    Metadata m_meta;       /* Patient name, patient id, etc. */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
					  coordinates */
    char m_xio_dose_input[_MAX_PATH];  /* Input XiO dose file to use as 
					  template for XiO dose saving. */
public:
    Rtds ();
    ~Rtds ();
    void load_dicom_dir (const char *dicom_dir);
    void load_xio (const char *xio_dir,	Slice_index *rdd);
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
