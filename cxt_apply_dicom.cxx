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
#include "cxt_io.h"
#include "gdcm_rtss.h"
#include "gdcm_series.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"

void
cxt_apply_dicom_dir (Cxt_structure_list *structures, char *dicom_dir)
{
    int i, j;
    Gdcm_series gs;
    std::string tmp;

    if (!dicom_dir) {
	return;
    }

    gs.load (dicom_dir);
    gs.get_best_ct ();
    if (!gs.m_have_ct) {
	return;
    }
    gdcm::File* file = gs.get_ct_slice ();

    /* Add geometry */
    int d;
    structures->have_geometry = 1;
    for (d = 0; d < 3; d++) {
	structures->offset[d] = gs.m_origin[d];
	structures->dim[d] = gs.m_dim[d];
	structures->spacing[d] = gs.m_spacing[d];
    }

    /* PatientName */
    tmp = file->GetEntryValue (0x0010, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_name = bfromcstr (tmp.c_str());
    }

    /* PatientID */
    tmp = file->GetEntryValue (0x0010, 0x0020);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_id = bfromcstr (tmp.c_str());
    }

    /* PatientSex */
    tmp = file->GetEntryValue (0x0010, 0x0040);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_sex = bfromcstr (tmp.c_str());
    }

    /* StudyID */
    tmp = file->GetEntryValue (0x0020, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->study_id = bfromcstr (tmp.c_str());
    }

    /* StudyInstanceUID */
    tmp = file->GetEntryValue (0x0020, 0x000d);
    structures->ct_study_uid = bfromcstr (tmp.c_str());

    /* SeriesInstanceUID */
    tmp = file->GetEntryValue (0x0020, 0x000e);
    structures->ct_series_uid = bfromcstr (tmp.c_str());
	
    /* FrameOfReferenceUID */
    tmp = file->GetEntryValue (0x0020, 0x0052);
    structures->ct_fref_uid = bfromcstr (tmp.c_str());

    /* slice numbers and slice uids */
    for (i = 0; i < structures->num_structures; i++) {
	Cxt_structure *curr_structure = &structures->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Cxt_polyline *curr_polyline = &curr_structure->pslist[j];
	    if (curr_polyline->num_vertices <= 0) {
		continue;
	    }
	    gs.get_slice_info (&curr_polyline->slice_no,
			       &curr_polyline->ct_slice_uid,
			       curr_polyline->z[0]);
	}
    }
}
