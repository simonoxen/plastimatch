/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_module.h"
#include "dcmtk_metadata.h"
#include "dicom_util.h"
#include "metadata.h"
#include "plm_uid_prefix.h"

void
Dcmtk_module_patient::set (DcmDataset *dataset, const Metadata::Pointer& meta)
{
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientName, "");
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientID, "");
    dataset->putAndInsertString (DCM_PatientBirthDate, "");
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientSex, "O");
}

void
Dcmtk_module_general_study::set (
    DcmDataset *dataset, 
    const Rt_study_metadata::Pointer& rsm)
{
    dataset->putAndInsertString (DCM_StudyInstanceUID, 
        rsm->get_study_uid());
    dataset->putAndInsertOFStringArray (DCM_StudyDate, 
        rsm->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_StudyTime, 
        rsm->get_study_time());
    dataset->putAndInsertString (DCM_ReferringPhysicianName, "");
    dcmtk_copy_from_metadata (dataset, rsm->get_study_metadata(), 
        DCM_StudyID, "");
    dataset->putAndInsertOFStringArray (DCM_AccessionNumber, "");
    dcmtk_copy_from_metadata (dataset, rsm->get_study_metadata (),
        DCM_StudyDescription, "");
}

void
Dcmtk_module_general_series::set_sro (
    DcmDataset *dataset, 
    const Rt_study_metadata::Pointer& rsm)
{
    dataset->putAndInsertOFStringArray (DCM_Modality, "REG");
    dataset->putAndInsertString (DCM_SeriesInstanceUID, 
        dicom_uid(PLM_UID_PREFIX).c_str());
    dataset->putAndInsertString (DCM_SeriesNumber, "");
}

