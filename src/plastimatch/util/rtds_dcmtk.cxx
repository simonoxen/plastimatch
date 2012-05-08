/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"

#include "plmutil.h"

#include "compiler_warnings.h"
#if PLM_DCM_USE_DCMTK
#include "dcmtk_loader.h"
#include "dcmtk_save.h"
#endif

void
Rtds::load_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_loader dss (dicom_dir);

    dss.parse_directory ();

#if defined (GCS_FIX)
    if (ds_rtss) {
        ds_rtss->rtss_load (rtds);
    }

    if (ds_rtdose) {
        ds_rtdose->rtdose_load (rtds);
    }
#endif

    printf ("Done.\n");
#endif
}

void
Rtds::save_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_save ds;

    ds.set_image (this->m_img);
    if (this->m_rtss && this->m_rtss->m_cxt) {
        ds.set_cxt (this->m_rtss->m_cxt, &this->m_rtss->m_meta);
    }
    ds.set_dose (this->m_dose);

    ds.save (dicom_dir);
#endif
}
