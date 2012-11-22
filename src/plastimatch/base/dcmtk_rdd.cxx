/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include "dcmtk_loader.h"
#include "plm_math.h"
#include "plm_uid_prefix.h"
#include "slice_index.h"

void
dcmtk_load_rdd (
    Slice_index *rdd,
    const char *dicom_dir
)
{
    if (!dicom_dir) {
	return;
    }

    Dcmtk_loader dss (dicom_dir);
    dss.parse_directory ();

#if defined (commentout)
    Gdcm_series gs;
    std::string tmp;

    gs.load (dicom_dir);
    gs.digest_files ();
    if (!gs.m_have_ct) {
	return;
    }
    gdcm::File* file = gs.get_ct_slice ();

    /* Add geometry */
    int d;
    float offset[3], spacing[3];
    rdd->m_loaded = 1;
    /* Convert double to float */
    for (d = 0; d < 3; d++) {
	offset[d] = gs.m_origin[d];
	spacing[d] = gs.m_spacing[d];
    }
    rdd->m_pih.set_from_gpuit (gs.m_dim, offset, spacing, 0);

    /* PatientName */
    set_metadata_from_gdcm_file (&rdd->m_demographics, file, 0x0010, 0x0010);

    /* PatientID */
    set_metadata_from_gdcm_file (&rdd->m_demographics, file, 0x0010, 0x0020);

    /* PatientSex */
    set_metadata_from_gdcm_file (&rdd->m_demographics, file, 0x0010, 0x0040);

    /* PatientPosition */
    set_metadata_from_gdcm_file (&rdd->m_demographics, file, 0x0018, 0x5100);

    /* StudyID */
    tmp = gdcm_file_GetEntryValue (file, 0x0020, 0x0010);
    if (tmp != gdcm_file_GDCM_UNFOUND()) {
	rdd->m_study_id = tmp.c_str();
    }

    /* StudyInstanceUID */
    tmp = gdcm_file_GetEntryValue (file, 0x0020, 0x000d);
    rdd->m_ct_study_uid = tmp.c_str();

    /* SeriesInstanceUID */
    tmp = gdcm_file_GetEntryValue (file, 0x0020, 0x000e);
    rdd->m_ct_series_uid = tmp.c_str();
	
    /* FrameOfReferenceUID */
    tmp = gdcm_file_GetEntryValue (file, 0x0020, 0x0052);
    rdd->m_ct_fref_uid = tmp.c_str();

    /* Slice uids */
    gs.get_slice_uids (&rdd->m_ct_slice_uids);
#endif
}
