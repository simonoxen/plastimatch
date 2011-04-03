/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plm_config.h"

#include "cxt_io.h"
#include "img_metadata.h"
#include "plm_image.h"
#include "referenced_dicom_dir.h"
#include "xio_ct.h"

class Gdcm_series;
class Rtss;

/* rtds = RT data set */
class plastimatch1_EXPORT Rtds {
public:
    Plm_image *m_img;                  /* CT image */
    Rtss *m_ss_image;                  /* RT structure set */
    Plm_image *m_dose;                 /* RT dose */

    Gdcm_series *m_gdcm_series;        /* Input dicom parse info */
    Referenced_dicom_dir m_rdd;        /* UIDs, etc */
    Img_metadata m_img_metadata;       /* Patient name, patient id, etc. */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
					  coordinates */
    char m_xio_dose_input[_MAX_PATH];  /* Input XiO dose file to use as 
					  template for XiO dose saving. */
public:
    Rtds ();
    ~Rtds ();
    void load_dicom_dir (const char *dicom_dir);
    void load_xio (
	const char *xio_dir,
	const char *dicom_dir,
	Plm_image_patient_position patient_pos
    );
    void load_ss_img (const char *ss_img, const char *ss_list);
    void load_dose_img (const char *dose_img);
    void load_dose_xio (const char *dose_xio, 
	Plm_image_patient_position patient_pos);
    void load_dose_astroid (const char *dose_astroid, 
	Plm_image_patient_position patient_pos);
    void load_dose_mc (const char *dose_mc, 
	Plm_image_patient_position patient_pos);
    void load_rdd (const char *rdd);
    void load_dicom (const char *dicom_dir); 
    void save_dicom (const char *output_dir);
    void convert_ss_img_to_cxt (void);
};

#endif
