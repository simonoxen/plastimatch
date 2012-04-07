/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_image.h"
#include "dcmtk_rtss.h"
#include "dcmtk_save.h"
#include "dcmtk_series_set.h"
#include "dcmtk_uid.h"
#include "file_util.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "rtds.h"

void
dcmtk_rtds_save (Rtds *rtds, const char *dicom_dir)
{
    Dcmtk_study_writer dsw;
    DcmDate::getCurrentDate (dsw.date_string);
    DcmTime::getCurrentTime (dsw.time_string);
    plm_generate_dicom_uid (dsw.study_uid, PLM_UID_PREFIX);
    plm_generate_dicom_uid (dsw.for_uid, PLM_UID_PREFIX);
    plm_generate_dicom_uid (dsw.ct_series_uid, PLM_UID_PREFIX);
    plm_generate_dicom_uid (dsw.rtss_uid, PLM_UID_PREFIX);

    if (rtds->m_img) {
        dcmtk_image_save (&dsw, rtds, dicom_dir);
    }
    if (rtds->m_rtss) {
        dcmtk_rtss_save (&dsw, rtds, dicom_dir);
    }
}
