/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "demographics.h"
#include "plm_image.h"
#include "xio_ct.h"

/* rtds = RT data set */
class Rtds {
public:
    Plm_image *m_img;                    /* CT image */
    Cxt_structure_list *m_cxt;           /* Structure set in polyline form */
    Plm_image *m_ss_img;                 /* Structure set in bitmap form */
    Cxt_structure_list *m_ss_list;       /* Names of structures/bitmap form */
    Plm_image *m_labelmap;               /* Structure set in bitmap form */
    Plm_image *m_dose;                   /* RT dose */
    char m_xio_dose_input[_MAX_PATH];    /* Input XiO dose file to use as 
					    template for XiO dose saving. */
    Xio_ct_transform *m_xio_transform;    /* Transformation from XiO to DICOM
					    coordinates */
    Demographics demographics;

public:
    Rtds () {
	int i;

	m_img = 0;
	m_ss_img = 0;
	m_ss_list = 0;
	m_labelmap = 0;
	m_cxt = 0;
	m_dose = 0;
	strcpy(m_xio_dose_input, "\0");

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
	if (m_cxt) {
	    cxt_destroy (m_cxt);
	}
	if (m_ss_img) {
	    delete m_ss_img;
	}
	if (m_ss_list) {
	    cxt_destroy (m_ss_list);
	}
	if (m_labelmap) {
	    delete m_labelmap;
	}
	if (m_dose) {
	    delete m_dose;
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
    void load_dose_xio (const char *dose_xio);
    void load_dose_astroid (const char *dose_astroid);
    void load_dose_mc (const char *dose_mc);
    void save_dicom (const char *dicom_dir);
    void convert_ss_img_to_cxt (void);
};

#if defined __cplusplus
extern "C" {
#endif


#if defined __cplusplus
}
#endif

#endif
