/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_metadata.h"
#include "dcmtk_rt_study.h"
#include "dcmtk_rt_study_p.h"
#include "dcmtk_series.h"
#include "dcmtk_slice_data.h"
#include "dcmtk_uid.h"
#include "rt_study_metadata.h"
#include "file_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "volume.h"

static void
dcmtk_save_slice (const Rt_study_metadata::Pointer drs, Dcmtk_slice_data *dsd)
{
    Pstring tmp;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();
    const Metadata *image_metadata = 0;
    if (drs) {
        image_metadata = drs->get_image_metadata ();
    }

    dataset->putAndInsertString (DCM_ImageType, 
        "DERIVED\\SECONDARY\\REFORMATTED");
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
        drs->get_study_date());
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
        drs->get_study_time());
    dataset->putAndInsertString (DCM_SOPClassUID, UID_CTImageStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, dsd->slice_uid);
    dataset->putAndInsertOFStringArray (DCM_StudyDate, drs->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_StudyTime, drs->get_study_time());
    dataset->putAndInsertString (DCM_AccessionNumber, "");
    dcmtk_copy_from_metadata (dataset, image_metadata, DCM_Modality, "CT");
    dataset->putAndInsertString (DCM_Manufacturer, "Plastimatch");
    dataset->putAndInsertString (DCM_ReferringPhysicianName, "");
    dcmtk_copy_from_metadata (dataset, image_metadata, 
        DCM_SeriesDescription, "");
    dcmtk_copy_from_metadata (dataset, image_metadata, DCM_PatientName, "");
    dcmtk_copy_from_metadata (dataset, image_metadata, DCM_PatientID, "");
    dcmtk_copy_from_metadata (dataset, image_metadata, 
        DCM_PatientBirthDate, "");
    dcmtk_copy_from_metadata (dataset, image_metadata, DCM_PatientSex, "");
    dataset->putAndInsertString (DCM_SliceThickness, dsd->sthk.c_str());
    dataset->putAndInsertString (DCM_SoftwareVersions, 
        PLASTIMATCH_VERSION_STRING);
    dcmtk_copy_from_metadata (dataset, image_metadata, 
        DCM_PatientPosition, "HFS");
    dataset->putAndInsertString (DCM_StudyInstanceUID, drs->get_study_uid());
    dataset->putAndInsertString (DCM_SeriesInstanceUID, 
        drs->get_ct_series_uid());
    dcmtk_copy_from_metadata (dataset, image_metadata, DCM_StudyID, "10001");
    dataset->putAndInsertString (DCM_SeriesNumber, "303");
    tmp.format ("%d", dsd->instance_no);
    dataset->putAndInsertString (DCM_InstanceNumber, tmp.c_str());
        //dataset->putAndInsertString (DCM_InstanceNumber, "0");
    /* DCM_PatientOrientation seems to be not required.  */
    // dataset->putAndInsertString (DCM_PatientOrientation, "L\\P");
    dataset->putAndInsertString (DCM_ImagePositionPatient, dsd->ipp);
    dataset->putAndInsertString (DCM_ImageOrientationPatient, dsd->iop);
    dataset->putAndInsertString (DCM_FrameOfReferenceUID, 
        drs->get_frame_of_reference_uid());
    /* XVI 4.5 requires a DCM_PositionReferenceIndicator */
    dataset->putAndInsertString (DCM_PositionReferenceIndicator, "SP");
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

    dataset->putAndInsertString (DCM_RescaleIntercept, "-1024");
    dataset->putAndInsertString (DCM_RescaleSlope, "1");
    dataset->putAndInsertString (DCM_RescaleType, "HU");

    dataset->putAndInsertString (DCM_WindowCenter, "40");
    dataset->putAndInsertString (DCM_WindowWidth, "400");

    /* Convert to 16-bit signed int */
    for (size_t i = 0; i < dsd->slice_size; i++) {
        float f = dsd->slice_float[i];
        dsd->slice_int16[i] = (int16_t) (f + 1024);
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
Dcmtk_rt_study::save_image (
    const char *dicom_dir)
{
    Dcmtk_slice_data dsd;
    dsd.vol = this->get_image_volume_float();
    dsd.slice_size = dsd.vol->dim[0] * dsd.vol->dim[1];
    dsd.slice_int16 = new int16_t[dsd.slice_size];
    float *dc = dsd.vol->direction_cosines.get();
    dsd.iop.format ("%f\\%f\\%f\\%f\\%f\\%f",
        dc[0], dc[1], dc[2], dc[3], dc[4], dc[5]);

    Plm_image_header pih (dsd.vol.get());
    d_ptr->dicom_metadata->set_image_header (pih);

    for (plm_long k = 0; k < dsd.vol->dim[2]; k++) {
        /* GCS FIX: direction cosines */
        float z_loc = dsd.vol->offset[2] + k * dsd.vol->spacing[2];
        dsd.instance_no = k;
        dsd.fn.format ("%s/image%03d.dcm", dicom_dir, (int) k);
        make_directory_recursive (dsd.fn);
        dsd.sthk.format ("%f", dsd.vol->spacing[2]);
        dsd.sloc.format ("%f", z_loc);
        dsd.ipp.format ("%f\\%f\\%f", dsd.vol->offset[0], dsd.vol->offset[1], 
            dsd.vol->offset[2] + k * dsd.vol->spacing[2]);
        dcmtk_uid (dsd.slice_uid, PLM_UID_PREFIX);

        dsd.slice_float = &((float*)dsd.vol->img)[k*dsd.slice_size];
        dcmtk_save_slice (d_ptr->dicom_metadata, &dsd);

        d_ptr->dicom_metadata->set_slice_uid (k, dsd.slice_uid);
    }
    delete[] dsd.slice_int16;
    d_ptr->dicom_metadata->set_slice_list_complete ();
}
