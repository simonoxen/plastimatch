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
    Gdcm_series gs;

    if (!dicom_dir) {
	return;
    }
    gs.load (dicom_dir);
    gs.digest_files ();

    if (gs.m_rtdose_file_list) {
	gdcm::File *file = (*(gs.m_rtdose_file_list))[0];
	const std::string& filename = file->GetFileName();
	rtds->m_dose = gdcm_dose_load (0, filename.c_str(), dicom_dir);
    }
    if (gs.m_rtstruct_file_list) {
	gdcm::File *file = (*(gs.m_rtstruct_file_list))[0];
	const std::string& filename = file->GetFileName();
	rtds->m_ss_image = new Ss_image;
	rtds->m_ss_image->load_gdcm_rtss (filename.c_str(), dicom_dir);
    }
}

void
rtds_patient_pos_from_dicom_dir (Rtds *rtds, const char *dicom_dir)
{
    Gdcm_series gs;
    std::string tmp;

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
	rtds->m_img->m_patient_pos = plm_image_patient_position_parse (tmp.c_str());
    } else {
	rtds->m_img->m_patient_pos =  PATIENT_POSITION_UNKNOWN;
    }
}
