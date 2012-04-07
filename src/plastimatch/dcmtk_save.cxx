/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_image.h"
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
    std::vector<Dcmtk_slice_data> slice_data;
    if (rtds->m_img) {
        dcmtk_image_save (&slice_data, rtds, dicom_dir);
    }
    if (rtds->m_ss_image) {
        dcmtk_rtss_save (&slice_data, rtds, dicom_dir);
    }
}
