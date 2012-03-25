/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_load.h"
#include "dcmtk_series_set.h"
#include "dcmtk_uid.h"
#include "file_util.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "rtds.h"

typedef
struct dcmtk_slice_data
{
    Pstring fn;
    Rtds *rtds;
    Volume *vol;

    size_t slice_size;
    float *slice_float;
    int16_t *slice_int16;

    OFString date_string;
    OFString time_string;
    char study_uid[100];
    char series_uid[100];
    char for_uid[100];
    Pstring ipp;
    Pstring iop;
    Pstring sloc;
    Pstring sthk;
} Dcmtk_slice_data;

void
dcmtk_save_slice (Dcmtk_slice_data *dsd)
{
    char uid[100];
    Pstring tmp;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    dataset->putAndInsertString (DCM_SOPClassUID, 
        UID_SecondaryCaptureImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, 
        dcmGenerateUniqueIdentifier (uid, SITE_INSTANCE_UID_ROOT));

    dataset->putAndInsertString (DCM_ImageType, 
        "DERIVED\\SECONDARY\\REFORMATTED");
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
        dsd->date_string);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
        dsd->time_string);
    dataset->putAndInsertString (DCM_SOPClassUID, UID_CTImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, 
        plm_generate_dicom_uid (uid, PLM_UID_PREFIX));
    dataset->putAndInsertOFStringArray (DCM_StudyDate, dsd->date_string);
    dataset->putAndInsertOFStringArray (DCM_StudyTime, dsd->time_string);
    dataset->putAndInsertString (DCM_AccessionNumber, "");
    dataset->putAndInsertString (DCM_Modality, "CT");
    dataset->putAndInsertString (DCM_Manufacturer, "Plastimatch");
    dataset->putAndInsertString (DCM_ReferringPhysicianName, "");
    dataset->putAndInsertString (DCM_PatientName, "");
    dataset->putAndInsertString (DCM_PatientID, "");
    dataset->putAndInsertString (DCM_PatientBirthDate, "");
    dataset->putAndInsertString (DCM_PatientSex, "");
    dataset->putAndInsertString (DCM_SliceThickness, dsd->sthk.c_str());
    dataset->putAndInsertString (DCM_SoftwareVersions,
        PLASTIMATCH_VERSION_STRING);
    /* GCS FIX: PatientPosition */
    dataset->putAndInsertString (DCM_PatientPosition, "HFS");
    dataset->putAndInsertString (DCM_StudyInstanceUID, dsd->study_uid);
    dataset->putAndInsertString (DCM_SeriesInstanceUID, dsd->series_uid);
    dataset->putAndInsertString (DCM_StudyID, "10001");
    dataset->putAndInsertString (DCM_SeriesNumber, "303");
    dataset->putAndInsertString (DCM_InstanceNumber, "0");
    /* GCS FIX: PatientOrientation */
    dataset->putAndInsertString (DCM_PatientOrientation, "L/P");
    dataset->putAndInsertString (DCM_ImagePositionPatient, dsd->ipp);
    dataset->putAndInsertString (DCM_ImageOrientationPatient, dsd->iop);
    dataset->putAndInsertString (DCM_FrameOfReferenceUID, dsd->for_uid);
    dataset->putAndInsertString (DCM_SliceLocation, dsd->sloc.c_str());
    dataset->putAndInsertString (DCM_SamplesPerPixel, "1");
    dataset->putAndInsertString (DCM_PhotometricInterpretation, "MONOCHROME2");
    dataset->putAndInsertUint16 (DCM_Rows, (Uint16) dsd->vol->dim[1]);
    dataset->putAndInsertUint16 (DCM_Columns, (Uint16) dsd->vol->dim[0]);
    tmp.format ("%f\\%f", dsd->vol->spacing[0], dsd->vol->spacing[1]);
    dataset->putAndInsertString (DCM_PixelSpacing, tmp.c_str());
    dataset->putAndInsertString (DCM_BitsAllocated, "16");
    dataset->putAndInsertString (DCM_BitsStored, "16");
    dataset->putAndInsertString (DCM_HighBit, "15");
    dataset->putAndInsertString (DCM_PixelRepresentation, "1");
    dataset->putAndInsertString (DCM_RescaleIntercept, "0");
    dataset->putAndInsertString (DCM_RescaleSlope, "1");
    dataset->putAndInsertString (DCM_RescaleType, "US");

    /* Convert to 16-bit signed int */
    for (size_t i = 0; i < dsd->slice_size; i++) {
        float f = dsd->slice_float[i];
        dsd->slice_int16[i] = (int16_t) f;
    }

    dataset->putAndInsertUint16Array (DCM_PixelData, 
        (Uint16*) dsd->slice_int16, dsd->slice_size);
    OFCondition status = fileformat.saveFile (dsd->fn.c_str(), 
        EXS_LittleEndianExplicit);
    if (status.bad()) {
        print_and_exit ("Error: cannot write DICOM file (%s)\n", 
            status.text());
    }
}

void
dcmtk_save_image (Rtds *rtds, const char *dicom_dir)
{
    Dcmtk_slice_data dsd;
    DcmDate::getCurrentDate (dsd.date_string);
    DcmTime::getCurrentTime (dsd.time_string);
    dsd.rtds = rtds;
    dsd.vol = rtds->m_img->gpuit_float();

    dsd.slice_size = dsd.vol->dim[0] * dsd.vol->dim[1];
    dsd.slice_int16 = new int16_t[dsd.slice_size];
    float *dc = dsd.vol->direction_cosines.m_direction_cosines;
    dsd.iop.format ("%f\\%f\\%f\\%f\\%f\\%f",
        dc[0], dc[1], dc[2], dc[3], dc[4], dc[5]);

    plm_generate_dicom_uid (dsd.study_uid, PLM_UID_PREFIX);
    plm_generate_dicom_uid (dsd.series_uid, PLM_UID_PREFIX);
    plm_generate_dicom_uid (dsd.for_uid, PLM_UID_PREFIX);

    for (plm_long k = 0; k < dsd.vol->dim[2]; k++) {
        dsd.fn.format ("%s/image%03d.dcm", dicom_dir, (int) k);
        make_directory_recursive (dsd.fn);
        /* GCS FIX: direction cosines */
        dsd.sthk.format ("%f", dsd.vol->spacing[2]);
        dsd.sloc.format ("%f", dsd.vol->offset[2] + k * dsd.vol->spacing[2]);
        dsd.ipp.format ("%f\\%f\\%f", dsd.vol->offset[0], dsd.vol->offset[1], 
            dsd.vol->offset[2] + k * dsd.vol->spacing[2]);

        dsd.slice_float = &((float*)dsd.vol->img)[k*dsd.slice_size];
        dcmtk_save_slice (&dsd);
    }
    delete[] dsd.slice_int16;
}

void
dcmtk_save_rtds (Rtds *rtds, const char *dicom_dir)
{
    if (rtds->m_img) {
        dcmtk_save_image (rtds, dicom_dir);
    }
}
