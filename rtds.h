/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "demographics.h"
#include "plm_image.h"
#include "referenced_dicom_dir.h"
#include "ss_image.h"
#include "xio_ct.h"

/* rtds = RT data set */
class Rtds {
public:
    Plm_image *m_img;                  /* CT image */
    Ss_image *m_ss_image;              /* Structure set lossless bitmap form */
    Plm_image *m_dose;                 /* RT dose */
    char m_xio_dose_input[_MAX_PATH];  /* Input XiO dose file to use as 
					    template for XiO dose saving. */
    Xio_ct_transform *m_xio_transform; /* Transformation from XiO to DICOM
					    coordinates */
    Demographics demographics;         /* Patient name, patient id, etc. */
    Referenced_dicom_dir *m_rdd;       /* UIDs for SS output */

public:
    Rtds () {
	int i;

	m_img = 0;
	m_ss_image = 0;
	m_dose = 0;
	m_rdd = 0;
	strcpy (m_xio_dose_input, "\0");

	m_xio_transform = (Xio_ct_transform*) malloc (sizeof (Xio_ct_transform));
	m_xio_transform->patient_pos = PATIENT_POSITION_UNKNOWN;
	m_xio_transform->x_offset = 0;
	m_xio_transform->y_offset = 0;
	for (i = 0; i <= 8; i++) {
	    m_xio_transform->direction_cosines[i] = 0;
	}
	m_xio_transform->direction_cosines[0] = 1;
	m_xio_transform->direction_cosines[4] = 1;
	m_xio_transform->direction_cosines[8] = 1;
    }
    ~Rtds () {
	if (m_img) {
	    delete m_img;
	}
	if (m_ss_image) {
	    delete m_ss_image;
	}
	if (m_dose) {
	    delete m_dose;
	}
	if (m_rdd) {
	    delete m_rdd;
	}
    }
    plastimatch1_EXPORT
    void load_dicom_dir (const char *dicom_dir);
    void load_xio (
	const char *xio_dir,
	const char *dicom_dir,
	Plm_image_patient_position patient_pos
    );
    plastimatch1_EXPORT
    void load_ss_img (const char *ss_img, const char *ss_list);
    void load_dose_img (const char *dose_img);
    void load_dose_xio (const char *dose_xio, Plm_image_patient_position patient_pos);
    void load_dose_astroid (const char *dose_astroid, Plm_image_patient_position patient_pos);
    void load_dose_mc (const char *dose_mc, Plm_image_patient_position patient_pos);
    void load_rdd (const char *rdd);
    void save_dicom (const char *output_dir);
    void convert_ss_img_to_cxt (void);
};

#if defined __cplusplus
extern "C" {
#endif


#if defined __cplusplus
}
#endif

#endif
