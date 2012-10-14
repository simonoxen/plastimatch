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
#include "dcmtk_loader.h"
#include "dcmtk_metadata.h"
#include "dcmtk_save.h"
#include "dcmtk_series.h"
#include "file_util.h"
#include "logfile.h"
#include "plm_image.h"
#include "plm_math.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "volume.h"

/* This is the tolerance on irregularity of the grid spacing (in mm) */
#define GFOV_SPACING_TOL (1e-1)

template <class T> 
void
dcmtk_dose_copy (float *img_out, T *img_in, int nvox, float scale)
{
    for (int i = 0; i < nvox; i++) {
	img_out[i] = img_in[i] * scale;
    }
}

void
Dcmtk_loader::rtdose_load ()
{
    int rc;
    const char *val;
    uint16_t val_u16;
    plm_long dim[3];
    float ipp[3];
    float spacing[3];
    float *gfov;    /* gfov = GridFrameOffsetVector */
    plm_long gfov_len;
    const char *gfov_str;

    /* Modality -- better be RTDOSE */
    std::string modality = this->ds_rtdose->get_modality();
    if (modality == "RTDOSE") {
        printf ("Trying to load rt dose.\n");
    } else {
        print_and_exit ("Oops.\n");
    }

    /* FIX: load metadata such as patient name, etc. */

    /* ImagePositionPatient */
    val = this->ds_rtdose->get_cstr (DCM_ImagePositionPatient);
    if (!val) {
        print_and_exit ("Couldn't find DCM_ImagePositionPatient in rtdose\n");
    }
    rc = sscanf (val, "%f\\%f\\%f", &ipp[0], &ipp[1], &ipp[2]);
    if (rc != 3) {
	print_and_exit ("Error parsing RTDOSE ipp.\n");
    }

    /* Rows */
    if (!this->ds_rtdose->get_uint16 (DCM_Rows, &val_u16)) {
        print_and_exit ("Couldn't find DCM_Rows in rtdose\n");
    }
    dim[1] = val_u16;

    /* Columns */
    if (!this->ds_rtdose->get_uint16 (DCM_Columns, &val_u16)) {
        print_and_exit ("Couldn't find DCM_Columns in rtdose\n");
    }
    dim[0] = val_u16;

    /* PixelSpacing */
    val = this->ds_rtdose->get_cstr (DCM_PixelSpacing);
    if (!val) {
        print_and_exit ("Couldn't find DCM_PixelSpacing in rtdose\n");
    }
    rc = sscanf (val, "%g\\%g", &spacing[1], &spacing[0]);
    if (rc != 2) {
	print_and_exit ("Error parsing RTDOSE pixel spacing.\n");
    }

    /* GridFrameOffsetVector */
    val = this->ds_rtdose->get_cstr (DCM_GridFrameOffsetVector);
    if (!val) {
        print_and_exit ("Couldn't find DCM_GridFrameOffsetVector in rtdose\n");
    }
    gfov = 0;
    gfov_len = 0;
    gfov_str = val;
    while (1) {
	int len;
	gfov = (float*) realloc (gfov, (gfov_len + 1) * sizeof(float));
	rc = sscanf (gfov_str, "%g%n", &gfov[gfov_len], &len);
	if (rc != 1) {
	    break;
	}
	gfov_len ++;
	gfov_str += len;
	if (gfov_str[0] == '\\') {
	    gfov_str ++;
	}
    }
    dim[2] = gfov_len;
    if (gfov_len == 0) {
	print_and_exit ("Error parsing RTDOSE gfov.\n");
    }

    /* --- Analyze GridFrameOffsetVector --- */

    /* (1) Make sure first element is 0. */
    if (gfov[0] != 0.) {
	if (gfov[0] == ipp[2]) {
	    /* In this case, gfov values are absolute rather than relative 
	       positions, but we process the same way. */
	} else {
	    /* This is wrong.  But Nucletron does it. */
	    logfile_printf (
		"Warning: RTDOSE gfov[0] is neither 0 nor ipp[2].\n"
		"This violates the DICOM standard.  Proceeding anyway...\n");
	    /* Nucletron seems to work by ignoring absolute offset (???) */
	}
    }

    /* (2) Handle case where gfov_len == 1 (only one slice). */
    if (gfov_len == 1) {
	spacing[2] = spacing[0];
    }

    /* (3) Check to make sure spacing is regular. */
    for (plm_long i = 1; i < gfov_len; i++) {
	if (i == 1) {
	    spacing[2] = gfov[1] - gfov[0];
	} else {
	    float sp = gfov[i] - gfov[i-1];
	    if (fabs(sp - spacing[2]) > GFOV_SPACING_TOL) {
		print_and_exit ("Error RTDOSE grid has irregular spacing:"
		    "%f vs %f.\n", sp, spacing[2]);
	    }
	}
    }

    /* DoseGridScaling -- if element doesn't exist, scaling is 1.0 */
    float dose_scaling = 1.0;
    val = this->ds_rtdose->get_cstr (DCM_DoseGridScaling);
    if (val) {
        /* No need to check for success, let scaling be 1.0 if failure */
        sscanf (val, "%f", &dose_scaling);
    }

    printf ("RTDOSE: dim = %d %d %d\n        %f %f %f\n        %f %f %f\n",
        (int) dim[0], (int) dim[1], (int) dim[2],
        ipp[0], ipp[1], ipp[2], 
        spacing[0], spacing[1], spacing[2]);

    uint16_t bits_alloc, bits_stored, high_bit, pixel_rep;
    rc = this->ds_rtdose->get_uint16 (DCM_BitsAllocated, &bits_alloc);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_BitsAllocated in rtdose\n");
    }
    rc = this->ds_rtdose->get_uint16 (DCM_BitsStored, &bits_stored);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_BitsStored in rtdose\n");
    }
    rc = this->ds_rtdose->get_uint16 (DCM_HighBit, &high_bit);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_HighBit in rtdose\n");
    }
    rc = this->ds_rtdose->get_uint16 (DCM_PixelRepresentation, &pixel_rep);
    if (!rc) {
        print_and_exit ("Couldn't find DCM_PixelRepresentation in rtdose\n");
    }

    printf ("Bits_alloc: %d\n", (int) bits_alloc);
    printf ("Bits_stored: %d\n", (int) bits_stored);
    printf ("High_bit: %d\n", (int) high_bit);
    printf ("Pixel_rep: %d\n", (int) pixel_rep);

    /* Create output dose image */
    delete this->dose;
    this->dose = new Plm_image;

    /* Create Volume */
    Volume *vol = new Volume (dim, ipp, spacing, 0, PT_FLOAT, 1);
    float *img = (float*) vol->img;

    /* Bind volume to plm_image */
    this->dose->set_gpuit (vol);

    /* PixelData */
    unsigned long length = 0;
    if (pixel_rep == 0) {
        const uint16_t* pixel_data;
        rc = this->ds_rtdose->m_flist.front()->get_uint16_array (
            DCM_PixelData, &pixel_data, &length);
        printf ("rc = %d, length = %lu, npix = %ld\n", 
            rc, length, (long) vol->npix);
        if (bits_stored == 16) {
            dcmtk_dose_copy (img, (const uint16_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else if (bits_stored == 32) {
            dcmtk_dose_copy (img, (const uint32_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else {
            delete this->dose;
            this->dose = 0;
            print_and_exit ("Unknown pixel representation (%d %d)\n",
                bits_stored, pixel_rep);
        }
    } else {
        const int16_t* pixel_data;
        rc = this->ds_rtdose->m_flist.front()->get_int16_array (
            DCM_PixelData, &pixel_data, &length);
        if (bits_stored == 16) {
            dcmtk_dose_copy (img, (const int16_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else if (bits_stored == 32) {
            dcmtk_dose_copy (img, (const int32_t*) pixel_data, 
                vol->npix, dose_scaling);
        } else {
            delete this->dose;
            this->dose = 0;
            print_and_exit ("Unknown pixel representation (%d %d)\n",
                bits_stored, pixel_rep);
        }
    }
}

void
Dcmtk_save::save_dose (
    const Dcmtk_study_writer *dsw,
    const char *dicom_dir)
{
    OFCondition ofc;
    std::string s;

    /* Prepare output file */
    std::string filename = string_format ("%s/dose.dcm", dicom_dir);
    make_directory_recursive (filename);

    /* Prepare dcmtk */
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    dataset->putAndInsertString (DCM_ImageType, 
        "DERIVED\\SECONDARY\\REFORMATTED");
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
        dsw->date_string);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
        dsw->time_string);

    /* ----------------------------------------------------------------- */
    /*     Part 1  -- General header                                     */
    /* ----------------------------------------------------------------- */
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
        dsw->date_string);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
        dsw->time_string);
    dataset->putAndInsertOFStringArray(DCM_InstanceCreatorUID, 
        PLM_UID_PREFIX);
    dataset->putAndInsertString (DCM_SOPClassUID, UID_RTDoseStorage);
    dataset->putAndInsertString (DCM_SOPInstanceUID, dsw->dose_instance_uid);
    dataset->putAndInsertOFStringArray (DCM_StudyDate, dsw->date_string);
    dataset->putAndInsertOFStringArray (DCM_StudyTime, dsw->time_string);
    dataset->putAndInsertOFStringArray (DCM_AccessionNumber, "");
    dataset->putAndInsertOFStringArray (DCM_Modality, "RTDOSE");
    dataset->putAndInsertString (DCM_Manufacturer, "Plastimatch");
    dataset->putAndInsertString (DCM_InstitutionName, "");
    dataset->putAndInsertString (DCM_ReferringPhysicianName, "");
    dataset->putAndInsertString (DCM_StationName, "");
    dcmtk_put_metadata (dataset, this->dose_meta, DCM_SeriesDescription, "");
    dataset->putAndInsertString (DCM_ManufacturerModelName, "Plastimatch");
    dcmtk_put_metadata (dataset, this->dose_meta, DCM_PatientName, "");
    dcmtk_put_metadata (dataset, this->dose_meta, DCM_PatientID, "");
    dataset->putAndInsertString (DCM_PatientBirthDate, "");
    dcmtk_put_metadata (dataset, this->dose_meta, DCM_PatientSex, "O");
    dataset->putAndInsertString (DCM_SliceThickness, "");
    dataset->putAndInsertString (DCM_SoftwareVersions,
        PLASTIMATCH_VERSION_STRING);
    dataset->putAndInsertString (DCM_StudyInstanceUID, dsw->study_uid);
    dataset->putAndInsertString (DCM_SeriesInstanceUID, dsw->dose_series_uid);
    dcmtk_put_metadata (dataset, this->dose_meta, DCM_StudyID, "");
    dataset->putAndInsertString (DCM_SeriesNumber, "");
    dataset->putAndInsertString (DCM_InstanceNumber, "1");
#if defined (commentout)
    s = string_format ("%g\\%g\\%g", 
	this->dose->offset[0], this->dose->offset[1], this->dose->offset[2]);
    dataset->putAndInsertString (DCM_ImagePositionPatient, s.c_str());
#endif

    /* ----------------------------------------------------------------- */
    /*     Write the output file                                         */
    /* ----------------------------------------------------------------- */
    ofc = fileformat.saveFile (filename.c_str(), EXS_LittleEndianExplicit);
    if (ofc.bad()) {
        print_and_exit ("Error: cannot write DICOM RTDOSE (%s)\n", 
            ofc.text());
    }

#if defined (commentout)
    /* ----------------------------------------------------------------- */
    /*     Part 1  -- General header                                     */
    /* ----------------------------------------------------------------- */
    /* ImagePositionPatient */
    s = gdcm::Util::Format ("%g\\%g\\%g", 
	plh.m_origin[0], plh.m_origin[1], plh.m_origin[2]);
    gf->InsertValEntry (s, 0x0020, 0x0032);

    /* ImageOrientationPatient */
    s = gdcm::Util::Format ("%g\\%g\\%g\\%g\\%g\\%g",
	plh.m_direction[0][0],
	plh.m_direction[0][1],
	plh.m_direction[0][2],
	plh.m_direction[1][0],
	plh.m_direction[1][1],
	plh.m_direction[1][2]);
    gf->InsertValEntry (s, 0x0020, 0x0037);

    /* FrameOfReferenceUID */
    gf->InsertValEntry ((const char*) rdd->m_ct_fref_uid, 0x0020, 0x0052);

    /* SamplesPerPixel */
    gf->InsertValEntry ("1", 0x0028, 0x0002);
    /* PhotometricInterpretation */
    gf->InsertValEntry ("MONOCHROME2", 0x0028, 0x0004);
    /* NumberOfFrames */
    s = gdcm::Util::Format ("%d", plh.Size(2));
    gf->InsertValEntry (s, 0x0028, 0x0008);

    /* FrameIncrementPointer */
    /* Note: InsertValEntry doesn't work for AT value representations
       gf->InsertValEntry ("3004,000c", 0x0028, 0x0009); */
    uint16_t fip[2] = { 0x3004, 0x000c };
    gf->InsertBinEntry ((uint8_t*)fip, 4, 0x0028, 0x0009, std::string("AT"));

    /* Rows */
    s = gdcm::Util::Format ("%d", plh.Size(1));
    gf->InsertValEntry (s, 0x0028, 0x0010);
    /* Columns */
    s = gdcm::Util::Format ("%d", plh.Size(0));
    gf->InsertValEntry (s, 0x0028, 0x0011);
    /* PixelSpacing */
    s = gdcm::Util::Format ("%g\\%g", plh.m_spacing[1], plh.m_spacing[0]);
    gf->InsertValEntry (s, 0x0028, 0x0030);

    /* BitsAllocated */
    gf->InsertValEntry ("32", 0x0028, 0x0100);
    /* BitsStored */
    gf->InsertValEntry ("32", 0x0028, 0x0101);
    /* HighBit */
    gf->InsertValEntry ("31", 0x0028, 0x0102);
    /* PixelRepresentation */
    if (meta->get_metadata(0x3004, 0x0004) != "ERROR") {
	gf->InsertValEntry ("0", 0x0028, 0x0103);
    } else {
	gf->InsertValEntry ("1", 0x0028, 0x0103);
    }

    /* Do I need SmallestImagePixelValue, LargestImagePixelValue? */

    /* DoseUnits */
    gf->InsertValEntry ("GY", 0x3004, 0x0002);
    /* DoseType */
    if (meta->get_metadata(0x3004, 0x0004) != "") {
	set_gdcm_file_from_metadata (gf, meta, 0x3004, 0x0004);
    } else {
	gf->InsertValEntry ("PHYSICAL", 0x3004, 0x0004);
    }

    /* DoseSummationType */
    gf->InsertValEntry ("PLAN", 0x3004, 0x000a);

    /* GridFrameOffsetVector */
    s = std::string ("0");
    for (i = 1; i < plh.Size(2); i++) {
	s += gdcm::Util::Format ("\\%g", i * plh.m_spacing[2]);
    }
    gf->InsertValEntry (s, 0x3004, 0x000c);

    /* GCS FIX:
       Leave ReferencedRTPlanSequence empty (until I can cross reference) */

    /* We need to convert image to uint16_t, but first we need to 
       scale it.  The maximum dose needs to fit in a 16-bit unsigned 
       integer.  Older versions of plastimatch set the dose_scale to a 
       fixed value of 0.04 (based on the fact that this number was found 
       in the XiO sample).  With this scaling, the maximum dose is 262 Gy. 
       Now we compute an appropriate scaling factor based on the maximum 
       dose. */

    /* Copy the image so we don't corrupt the original */
    Plm_image *tmp = pli->clone ();
    /* Find the maximum value in the image */
    double min_val, max_val, avg;
    int non_zero, num_vox;
    tmp->convert (PLM_IMG_TYPE_ITK_FLOAT);
    itk_image_stats (tmp->m_itk_float, &min_val, &max_val, &avg, 
	&non_zero, &num_vox);

#ifndef UINT32_T_MAX
#define UINT32_T_MAX (0xffffffff)
#endif
#ifndef INT32_T_MAX
#define INT32_T_MAX (0x7fffffff)
#endif
#ifndef INT32_T_MIN
#define INT32_T_MIN (-0x7fffffff - 1)
#endif

    float dose_scale;

    if (meta->get_metadata(0x3004, 0x0004) != "ERROR") {
	/* Dose is unsigned integer */
	dose_scale = max_val / UINT32_T_MAX * 1.001;
    } else {
	/* Dose error is signed integer */
	float dose_scale_min = min_val / INT32_T_MIN * 1.001;
	float dose_scale_max = max_val / INT32_T_MAX * 1.001;
	dose_scale = std::max(dose_scale_min, dose_scale_max);
    }

    /* Scale the image */
    tmp->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
    Volume *vol = (Volume*) tmp->m_gpuit;
    volume_scale (vol, 1 / dose_scale);

    /* Convert to integer */
    if (meta->get_metadata(0x3004, 0x0004) != "ERROR") {
	tmp->convert (PLM_IMG_TYPE_GPUIT_UINT32);
    } else {
	tmp->convert (PLM_IMG_TYPE_GPUIT_INT32);
    }

    vol = (Volume*) tmp->m_gpuit;

    /* DoseGridScaling */
    s = gdcm::Util::Format ("%g", dose_scale);
    gf->InsertValEntry (s, 0x3004, 0x000e);
    /* PixelData */
    gdcm::FileHelper gfh (gf);
    gfh.SetImageData ((uint8_t*) vol->img, vol->npix * vol->pix_size);

    /* Do the actual writing out to file */
    gfh.WriteDcmExplVR (dose_fn);

    delete tmp;
#endif
}
