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
	rtds->m_cxt = cxt_create ();
	gdcm_rtss_load (rtds->m_cxt, filename.c_str(), dicom_dir);
    }
}
