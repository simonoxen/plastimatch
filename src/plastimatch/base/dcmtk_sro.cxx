/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_module_general_series.h"
#include "dcmtk_module_general_study.h"
#include "dcmtk_module_patient.h"
#include "dcmtk_sro.h"
#include "dicom_util.h"
#include "file_util.h"
#include "plm_uid_prefix.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "xform.h"

void
Dcmtk_sro::save (
    Xform* xf,
    const Rt_study_metadata::Pointer& rsm_src,
    const Rt_study_metadata::Pointer& rsm_reg,
    const std::string& dicom_dir)
{
    Xform xf_aff;
    xform_to_aff (&xf_aff, xf, 0);

    AffineTransformType::Pointer itk_aff = xf_aff.get_aff();

    /* Prepare output file */
    std::string sro_fn = string_format ("%s/sro.dcm", dicom_dir.c_str());
    make_directory_recursive (sro_fn);

    /* Prepare dcmtk */
    OFCondition ofc;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    Rt_study_metadata::Pointer rsm;
    Metadata *study_meta = 0;
    if (!rsm_src || !rsm_reg) {
        print_and_exit ("Sorry, anonymous spatial registration objects "
            "are not yet supported.\n");
    }

    /* Not sure about this... */
    rsm = rsm_src;
    study_meta = rsm_src->get_study_metadata ();

    /* Patient module, general study module */
    Dcmtk_module_patient::set (dataset, study_meta);
    Dcmtk_module_general_study::set (dataset, rsm);

    /* General series module */
    Dcmtk_module_general_series::set_sro (dataset, rsm);

    /* Spatial registration module */
    dataset->putAndInsertString (DCM_SOPClassUID, 
        UID_SpatialRegistrationStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, 
        dicom_uid(PLM_UID_PREFIX).c_str());
    dataset->putAndInsertOFStringArray (DCM_ContentDate, 
        rsm->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_ContentTime, 
        rsm->get_study_time());

    /* Spatial registration module -- fixed image */
    DcmItem *reg_item = 0;
    dataset->findOrCreateSequenceItem (
        DCM_RegistrationSequence, reg_item, -2);
    reg_item->putAndInsertString (
        DCM_FrameOfReferenceUID, 
        rsm_reg->get_frame_of_reference_uid());
    DcmItem *mr_item = 0;
    reg_item->findOrCreateSequenceItem (
        DCM_MatrixRegistrationSequence, mr_item, -2);
    DcmItem *rtc_item = 0;
    mr_item->findOrCreateSequenceItem (
        DCM_RegistrationTypeCodeSequence, rtc_item, -2);
    rtc_item->putAndInsertString (DCM_CodeValue, "125025");
    rtc_item->putAndInsertString (DCM_CodingSchemeDesignator, "DCM");
    rtc_item->putAndInsertString (DCM_CodeMeaning, "Visual Alignment");
    DcmItem *m_item = 0;
    mr_item->findOrCreateSequenceItem (DCM_MatrixSequence, m_item, -2);
    m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrix,
        "1.0\\0.0\\0.0\\0.0\\0.0\\1.0\\0.0\\0.0\\"
        "0.0\\0.0\\1.0\\0.0\\0.0\\0.0\\0.0\\1.0");
    m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrixType,
        "RIGID");

    /* Spatial registration module -- moving image */
    dataset->findOrCreateSequenceItem (
        DCM_RegistrationSequence, reg_item, -2);
    reg_item->putAndInsertString (
        DCM_FrameOfReferenceUID, 
        rsm_src->get_frame_of_reference_uid());
    reg_item->findOrCreateSequenceItem (
        DCM_MatrixRegistrationSequence, mr_item, -2);
    mr_item->findOrCreateSequenceItem (
        DCM_RegistrationTypeCodeSequence, rtc_item, -2);
    rtc_item->putAndInsertString (DCM_CodeValue, "125025");
    rtc_item->putAndInsertString (DCM_CodingSchemeDesignator, "DCM");
    rtc_item->putAndInsertString (DCM_CodeMeaning, "Visual Alignment");
    mr_item->findOrCreateSequenceItem (DCM_MatrixSequence, m_item, -2);
    std::string matrix_string;
    const AffineTransformType::MatrixType& itk_aff_mat 
        = itk_aff->GetMatrix ();
    const AffineTransformType::OutputVectorType& itk_aff_off 
        = itk_aff->GetOffset ();
    matrix_string = string_format (
        "%f\\%f\\%f\\0.0\\%f\\%f\\%f\\0.0\\"
        "%f\\%f\\%f\\0.0\\%f\\%f\\%f\\1.0",
        itk_aff_mat[0][0],
        itk_aff_mat[0][1],
        itk_aff_mat[0][2],
        itk_aff_mat[1][0],
        itk_aff_mat[1][1],
        itk_aff_mat[1][2],
        itk_aff_mat[2][0],
        itk_aff_mat[2][1],
        itk_aff_mat[2][2],
        itk_aff_off[0],
        itk_aff_off[1],
        itk_aff_off[2]);
    m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrix,
        matrix_string.c_str());
    m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrixType,
        "RIGID");

    /* ----------------------------------------------------------------- *
     *  Write the output file
     * ----------------------------------------------------------------- */
    ofc = fileformat.saveFile (sro_fn.c_str(), EXS_LittleEndianExplicit);
    if (ofc.bad()) {
        print_and_exit (
            "Error: cannot write DICOM Spatial Registration (%s)\n", 
            ofc.text());
    }
}
