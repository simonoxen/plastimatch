/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include "gdcmFile.h"

#include "gdcm_dose.h"
#include "gdcm_rtss.h"
#include "gdcm_series.h"
#include "gdcm_series_helper_2.h"
#include "logfile.h"
#include "plm_image_patient_position.h"
#include "print_and_exit.h"
#include "rtds_dicom.h"

//#if GDCM_MAJOR_VERSION < 2
//#include "gdcmUtil.h"
//#else
//#include "gdcmUIDGenerator.h"
//#endif

void
rtds_dicom_load (Rtds *rtds, const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

    if (rtds->m_gdcm_series) {
	delete rtds->m_gdcm_series;
    }
    rtds->m_gdcm_series = new Gdcm_series;
    rtds->m_gdcm_series->load (dicom_dir);
    rtds->m_gdcm_series->digest_files ();

    if (rtds->m_gdcm_series->m_rtdose_file_list) {
	gdcm::File *file = (*(rtds->m_gdcm_series->m_rtdose_file_list))[0];
	const std::string& filename = file->GetFileName();
	rtds->m_dose = gdcm_dose_load (0, filename.c_str(), dicom_dir);
    }
    if (rtds->m_gdcm_series->m_rtstruct_file_list) {
	gdcm::File *file = (*(rtds->m_gdcm_series->m_rtstruct_file_list))[0];
	const std::string& filename = file->GetFileName();
	rtds->m_ss_image = new Ss_image;
	rtds->m_ss_image->load_gdcm_rtss (filename.c_str());
    }

    /* Use existing itk reader for the image.
       This is required because the native dicom reader doesn't yet 
       handle things like MR. */
    rtds->m_img = plm_image_load_native (dicom_dir);
}

void
rtds_patient_pos_from_dicom_dir (Rtds *rtds, const char *dicom_dir)
{
    Gdcm_series gs;
    std::string tmp;
    Plm_image_patient_position patient_pos;

    if (!dicom_dir) {
	return;
    }

    gs.load (dicom_dir);
    gs.digest_files ();
    if (!gs.m_have_ct) {
	return;
    }
    gdcm::File* file = gs.get_ct_slice ();

    /* Get patient position */
    tmp = file->GetEntryValue (0x0018, 0x5100);
    if (tmp != gdcm::GDCM_UNFOUND) {
	patient_pos = plm_image_patient_position_parse (tmp.c_str());
    } else {
	patient_pos = PATIENT_POSITION_UNKNOWN;
    }

    if (rtds->m_img) rtds->m_img->m_patient_pos = patient_pos;
    if (rtds->m_dose) rtds->m_dose->m_patient_pos = patient_pos;
}
