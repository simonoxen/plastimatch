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
#include "plm_version.h"

void
Dcmtk_module::set_patient (
    DcmDataset *dataset, const Metadata::Pointer& meta)
{
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientName, "");
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientID, "");
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientBirthDate, "");
    dcmtk_copy_from_metadata (dataset, meta, DCM_PatientSex, "O");
}

void
Dcmtk_module::set_general_study (
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
    dataset->putAndInsertOFStringArray (DCM_StudyID, 
        rsm->get_study_id());
 }

void
Dcmtk_module::set_general_series_sro (
    DcmDataset *dataset, 
    const Rt_study_metadata::Pointer& rsm)
{
    dataset->putAndInsertOFStringArray (DCM_Modality, "REG");
    dataset->putAndInsertString (DCM_SeriesInstanceUID, 
        dicom_uid(PLM_UID_PREFIX).c_str());
    dataset->putAndInsertString (DCM_SeriesNumber, "");
}

void
Dcmtk_module::set_general_series (
    DcmDataset *dataset, 
    const Metadata::Pointer& meta)
{
    dcmtk_copy_from_metadata (dataset, meta, DCM_OperatorsName, "");
}

void
Dcmtk_module::set_rt_series (
    DcmDataset *dataset,
    const Metadata::Pointer& meta,
    const char* modality)
{
    dataset->putAndInsertOFStringArray (DCM_Modality, modality);
    /* Series Instance UID, this gets copied from e.g. 
        d_ptr->rt_study_metadata->get_dose_series_uid(), 
        in order to correctly make cross references between series.
        It is safe to set here, and allow caller to override. */
    dataset->putAndInsertString (DCM_SeriesInstanceUID, 
        dicom_uid(PLM_UID_PREFIX).c_str());
    dcmtk_copy_from_metadata (dataset, meta, DCM_SeriesNumber, "");
    dcmtk_copy_from_metadata (dataset, meta, DCM_SeriesDescription, "");
    /* Series Date, Series Time go here */
}

void
Dcmtk_module::set_general_equipment (DcmDataset *dataset,  
				     const Metadata::Pointer& meta)
{
  dcmtk_copy_from_metadata (dataset, meta, DCM_Manufacturer, "Plastimatch");
  dcmtk_copy_from_metadata (dataset, meta, DCM_InstitutionName, "");
  dcmtk_copy_from_metadata (dataset, meta, DCM_StationName, "");
  dcmtk_copy_from_metadata (dataset, meta, DCM_ManufacturerModelName, "Plastimatch");
  dcmtk_copy_from_metadata (dataset, meta, DCM_SoftwareVersions, PLASTIMATCH_VERSION_STRING);
}
