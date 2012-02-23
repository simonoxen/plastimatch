/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "rtds.h"

void
Rtds::load_dicom (const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

#if PLM_DCM_USE_DCMTK
    this->load_dcmtk (dicom_dir);
#else
    this->load_gdcm (dicom_dir);
#endif
}

void
Rtds::save_dicom (const char *dicom_dir)
{
    if (!dicom_dir) {
	return;
    }

#if PLM_DCM_USE_DCMTK
    this->save_dcmtk (dicom_dir);
#else
    this->save_gdcm (dicom_dir);
#endif
}
