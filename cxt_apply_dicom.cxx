/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"

#include "cxt_apply_dicom.h"
#include "gdcm_rtss.h"
#include "gdcm_series.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "rtss.h"

void
cxt_apply_dicom_dir (Rtss *cxt, const char *dicom_dir)
{
    int i, j;
    Gdcm_series gs;
    std::string tmp;

    if (!dicom_dir) {
	return;
    }

    gs.load (dicom_dir);
    gs.digest_files ();
    if (!gs.m_have_ct) {
	return;
    }
    gdcm::File* file = gs.get_ct_slice ();

    /* Add geometry */
    int d;
    cxt->have_geometry = 1;
    for (d = 0; d < 3; d++) {
	cxt->offset[d] = gs.m_origin[d];
	cxt->dim[d] = gs.m_dim[d];
	cxt->spacing[d] = gs.m_spacing[d];
    }

    /* PatientName */
    tmp = file->GetEntryValue (0x0010, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->m_demographics->m_patient_name = tmp.c_str();
    }

    /* PatientID */
    tmp = file->GetEntryValue (0x0010, 0x0020);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->m_demographics->m_patient_id = tmp.c_str();
    }

    /* PatientSex */
    tmp = file->GetEntryValue (0x0010, 0x0040);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->m_demographics->m_patient_sex = tmp.c_str();
    }

    /* StudyID */
    tmp = file->GetEntryValue (0x0020, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->study_id = tmp.c_str();
    }

    /* StudyInstanceUID */
    tmp = file->GetEntryValue (0x0020, 0x000d);
    cxt->ct_study_uid = tmp.c_str();

    /* SeriesInstanceUID */
    tmp = file->GetEntryValue (0x0020, 0x000e);
    cxt->ct_series_uid = tmp.c_str();
	
    /* FrameOfReferenceUID */
    tmp = file->GetEntryValue (0x0020, 0x0052);
    cxt->ct_fref_uid = tmp.c_str();

    /* slice numbers and slice uids */
    for (i = 0; i < cxt->num_structures; i++) {
	Rtss_structure *curr_structure = cxt->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Rtss_polyline *curr_polyline = curr_structure->pslist[j];
	    if (curr_polyline->num_vertices <= 0) {
		continue;
	    }
	    gs.get_slice_info (&curr_polyline->slice_no,
			       &curr_polyline->ct_slice_uid,
			       curr_polyline->z[0]);
	}
    }
}
