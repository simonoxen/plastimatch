/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>

#if GDCM_VERSION_1
#include "gdcm1_dose.h"
#include "gdcm1_series.h"
#endif
#include "plm_image.h"
#include "rtds.h"
#include "rtss.h"

void
Rtds::load_gdcm (const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

#if GDCM_VERSION_1
    Gdcm_series *gdcm_series = new Gdcm_series;
    gdcm_series->load (dicom_dir);
    gdcm_series->digest_files ();

     if (gdcm_series->m_rtdose_file_list) {
	const std::string& filename = gdcm_series->get_rtdose_filename();
	m_dose = gdcm1_dose_load (0, filename.c_str());
    }
    if (gdcm_series->m_rtstruct_file_list) {
	const std::string& filename = gdcm_series->get_rtstruct_filename();
	m_rtss = new Rtss (this);
	m_rtss->load_gdcm_rtss (filename.c_str(), m_rdd);
    }
#endif

    /* Use existing itk reader for the image.
       This is required because the native dicom reader doesn't yet 
       handle things like MR. */
    m_img = plm_image_load_native (dicom_dir);

#if GDCM_VERSION_1
    /* Use native reader to set meta */
    gdcm_series->get_metadata (this->get_metadata ());
    delete gdcm_series;
#endif
}

void
Rtds::save_gdcm (const char *output_dir)
{
    if (this->m_img) {
	printf ("Rtds::save_dicom: save_short_dicom()\n");
	this->m_img->save_short_dicom (output_dir, m_rdd, 
            this->get_metadata ());
    }
#if GDCM_VERSION_1
    if (this->m_rtss) {
	printf ("Rtds::save_dicom: save_gdcm_rtss()\n");
	this->m_rtss->save_gdcm_rtss (output_dir, m_rdd);
    }
    if (this->m_dose) {
	char fn[_MAX_PATH];
	printf ("Rtds::save_dicom: gdcm_save_dose()\n");
	snprintf (fn, _MAX_PATH, "%s/%s", output_dir, "dose.dcm");
	gdcm1_dose_save (m_dose, this->get_metadata(), m_rdd, fn);
    }
#endif
}
