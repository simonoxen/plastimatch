/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_module_general_study.h"
#include "dcmtk_metadata.h"
#include "metadata.h"

void
Dcmtk_module_general_study::set (DcmDataset *dataset, const Metadata* meta)
{
    /* These are stored in Dcmtk_rt_study class */
#if defined (commentout)
    dataset->putAndInsertString (DCM_StudyInstanceUID, 
        d_ptr->dicom_metadata->get_study_uid());
    dataset->putAndInsertOFStringArray (DCM_StudyDate, 
        d_ptr->dicom_metadata->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_StudyTime, 
        d_ptr->dicom_metadata->get_study_time());
    dataset->putAndInsertString (DCM_ReferringPhysicianName, "");
    dcmtk_copy_from_metadata (dataset, rtss_metadata, DCM_StudyID, "10001");
    dataset->putAndInsertOFStringArray (DCM_AccessionNumber, "");
#endif
}
