/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"

#include "compiler_warnings.h"
#if PLM_DCM_USE_DCMTK
#include "dcmtk_loader.h"
#include "dcmtk_save.h"
#endif
#include "plm_image.h"
#include "rtss.h"
#include "rtds.h"
#include "rtds_p.h"

void
Rtds::load_dcmtk (const char *dicom_path)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_loader dss (dicom_path);
    dss.set_rt_study (d_ptr->m_drs);
    dss.parse_directory ();

    this->m_img = dss.steal_plm_image ();
    Rtss_structure_set *rtss = dss.steal_rtss_structure_set ();
    if (rtss) {
	this->m_rtss = new Rtss (this);
        this->m_rtss->m_cxt = rtss;
    }
    d_ptr->m_dose = dss.steal_dose_image ();

    printf ("Done.\n");
#endif
}

void
Rtds::save_dcmtk (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_save ds;
    ds.set_rt_study (d_ptr->m_drs);

    ds.set_image (this->m_img);
    if (this->m_rtss && this->m_rtss->m_cxt) {
        ds.set_cxt (this->m_rtss->m_cxt);
    }
    if (d_ptr->m_dose) {
        ds.set_dose (d_ptr->m_dose->gpuit_float());
    }

    ds.save (dicom_dir);
#endif
}

void
Rtds::save_dcmtk_dose (const char *dicom_dir)
{
#if PLM_DCM_USE_DCMTK
    Dcmtk_save ds;
    ds.set_rt_study (d_ptr->m_drs);

    if (d_ptr->m_dose) {
        ds.set_dose (d_ptr->m_dose->gpuit_float());
    }

    ds.save (dicom_dir);
#endif
}
