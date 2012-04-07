/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "dcmtk_load.h"
#include "dcmtk_save.h"
#include "rtds.h"

void
Rtds::load_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    dcmtk_rtds_load (this, dicom_dir);
#endif
}

void
Rtds::save_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    dcmtk_rtds_save (this, dicom_dir);
#endif
}
