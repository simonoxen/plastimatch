/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"

#include "compiler_warnings.h"
#if PLM_DCM_USE_DCMTK
#include "dcmtk_loader.h"
#include "dcmtk_rt_study.h"
#endif
#include "plm_image.h"
#include "rtss.h"
#include "rt_study.h"
#include "rt_study_p.h"

void
Rt_study::load_dcmtk (const char *dicom_path)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_rt_study drs;
    drs.set_dicom_metadata (d_ptr->m_drs);
    drs.load (dicom_path);

    d_ptr->m_img = drs.get_image ();
    Rtss_structure_set::Pointer rtss = drs.get_rtss ();
    if (rtss) {
        d_ptr->m_rtss = Rtss::New ();
        d_ptr->m_rtss->set_structure_set (drs.get_rtss ());
    }
    d_ptr->m_dose = drs.get_dose ();
#endif
}

void
Rt_study::save_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_rt_study drs;
    drs.set_dicom_metadata (d_ptr->m_drs);
    drs.set_image (d_ptr->m_img);
    if (d_ptr->m_rtss) {
        drs.set_rtss (d_ptr->m_rtss->get_structure_set());
    }
    drs.set_dose (d_ptr->m_dose);
    //ds.generate_new_uids ();   // GCS FIX: Is this needed here?
    drs.save (dicom_dir);
#endif
}

void
Rt_study::save_dcmtk_dose (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_rt_study drs;
    drs.set_dicom_metadata (d_ptr->m_drs);
    drs.set_dose (d_ptr->m_dose);
    drs.save (dicom_dir);
#endif
}
