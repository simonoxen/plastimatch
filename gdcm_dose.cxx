/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmBinEntry.h"
#include "gdcmFile.h"
#include "gdcmFileHelper.h"
#include "gdcmGlobal.h"
#include "gdcmSeqEntry.h"
#include "gdcmSQItem.h"
#include "gdcmUtil.h"

#include "cxt.h"
#include "gdcm_dose.h"
#include "gdcm_series.h"
#include "plm_image.h"
#include "plm_image_type.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "volume.h"

/* winbase.h defines GetCurrentTime which conflicts with gdcm function */
#if defined GetCurrentTime
# undef GetCurrentTime
#endif

/* Gdcm has a broken header file gdcmCommon.h, which defines C99 types 
   (e.g. int32_t) when missing on MSVC.  However, it does so in an incorrect 
   way that conflicts with plm_int.h (which also fixes missing C99 types).  
   The workaround is to separately define the functions in flie_util.h 
   that we need. */
extern "C" {
gpuit_EXPORT
char* file_util_dirname (const char *filename);
gpuit_EXPORT
void make_directory_recursive (const char *dirname);
}

/* This is the tolerance on irregularity of the grid spacing (in mm) */
#define GFOV_SPACING_TOL (1e-1)

/* This function probes whether or not the file is a dicom dose format */
bool
gdcm_dose_probe (char *dose_fn)
{
    gdcm::File *gdcm_file = new gdcm::File;
    std::string tmp;

    gdcm_file->SetMaxSizeLoadEntry (0xffff);
    gdcm_file->SetFileName (dose_fn);
    gdcm_file->SetLoadMode (0);
    gdcm_file->Load();

    /* Modality -- better be RTDOSE */
    tmp = gdcm_file->GetEntryValue (0x0008, 0x0060);
    delete gdcm_file;
    if (strncmp (tmp.c_str(), "RTDOSE", strlen("RTDOSE"))) {
	return false;
    } else {
	return true;
    }
}

Plm_image*
gdcm_dose_load (Plm_image *pli, char *dose_fn, char *dicom_dir)
{
    int i, rc;
    gdcm::File *gdcm_file = new gdcm::File;
    Gdcm_series gs;
    std::string tmp;
    float ipp[3];
    int dim[3];
    float spacing[3];
    float *gfov;    /* gfov = GridFrameOffsetVector */
    int gfov_len;
    const char *gfov_str;
    float dose_scaling;

    gdcm_file->SetMaxSizeLoadEntry (0xffff);
    gdcm_file->SetFileName (dose_fn);
    gdcm_file->SetLoadMode (0);
    gdcm_file->Load();

    /* Modality -- better be RTSTRUCT */
    tmp = gdcm_file->GetEntryValue (0x0008, 0x0060);
    if (strncmp (tmp.c_str(), "RTDOSE", strlen("RTDOSE"))) {
	print_and_exit ("Error.  Input file not RTDOSE: %s\n",
	    dose_fn);
    }

    /* ImagePositionPatient */
    tmp = gdcm_file->GetEntryValue (0x0020, 0x0032);
    rc = sscanf (tmp.c_str(), "%f\\%f\\%f", &ipp[0], &ipp[1], &ipp[2]);
    if (rc != 3) {
	print_and_exit ("Error parsing RTDOSE ipp.\n");
    }

    /* Rows */
    tmp = gdcm_file->GetEntryValue (0x0028, 0x0010);
    rc = sscanf (tmp.c_str(), "%d", &dim[1]);
    if (rc != 1) {
	print_and_exit ("Error parsing RTDOSE rows.\n");
    }

    /* Columns */
    tmp = gdcm_file->GetEntryValue (0x0028, 0x0011);
    rc = sscanf (tmp.c_str(), "%d", &dim[0]);
    if (rc != 1) {
	print_and_exit ("Error parsing RTDOSE columns.\n");
    }

    /* PixelSpacing */
    tmp = gdcm_file->GetEntryValue (0x0028, 0x0030);
    rc = sscanf (tmp.c_str(), "%g\\%g", &spacing[0], &spacing[1]);
    if (rc != 2) {
	print_and_exit ("Error parsing RTDOSE pixel spacing.\n");
    }

    /* GridFrameOffsetVector */
    tmp = gdcm_file->GetEntryValue (0x3004, 0x000C);
    gfov = 0;
    gfov_len = 0;
    gfov_str = tmp.c_str();
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
	print_and_exit ("Error RTDOSE gfov[0] is not 0.\n");
    }

    /* (2) Handle case where gfov_len == 1 (only one slice). */
    if (gfov_len == 1) {
	spacing[2] = spacing[0];
    }

    /* (3) Check to make sure spacing is regular. */
    for (i = 1; i < gfov_len; i++) {
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

    /* DoseGridScaling */
    dose_scaling = 1.0;
    tmp = gdcm_file->GetEntryValue (0x3004, 0x000E);
    rc = sscanf (tmp.c_str(), "%f", &dose_scaling);
    /* If element doesn't exist, scaling is 1.0 */

    /* PixelData */
    gdcm::FileHelper gdcm_file_helper (gdcm_file);
    unsigned short* image_data 
	= (unsigned short*) gdcm_file_helper.GetImageData ();
    size_t image_data_size = gdcm_file_helper.GetImageDataSize();
    if (strcmp (gdcm_file->GetPixelType().c_str(), "16U")) {
	print_and_exit ("Error RTDOSE not type 16U (type=%s)\n",
	    gdcm_file->GetPixelType().c_str());
    }

    /* GCS FIX: Do I need to do something about endian-ness? */

    /* Create output pli if necessary */
    if (!pli) pli = new Plm_image;
    pli->free ();

    /* Create Volume */
    Volume *vol = volume_create (dim, ipp, spacing, PT_FLOAT, 0, 0);

    /* Copy data to volume */
    float *img = (float*) vol->img;
    for (i = 0; i < vol->npix; i++) {
	img[i] = image_data[i] * dose_scaling;
    }

    /* Bind volume to plm_image */
    pli->set_gpuit (vol);

#if defined (commentout)
    printf ("IPP = %f %f %f\n", ipp[0], ipp[1], ipp[2]);
    printf ("DIM = %d %d %d\n", dim[0], dim[1], dim[2]);
    printf ("SPC = %f %f %f\n", spacing[0], spacing[1], spacing[2]);
    printf ("NVX = %d\n", dim[0] * dim[1] * dim[2]);
    printf ("ID  = size %d, type %s\n", image_data_size, 
	gdcm_file->GetPixelType().c_str());
#endif

    free (gfov);
    delete gdcm_file;
    return pli;
}

void
gdcm_dose_save (Plm_image *pli, char *dose_fn)
{
    int i;
    gdcm::File *gf = new gdcm::File ();
    Gdcm_series gs;
    const std::string &current_date = gdcm::Util::GetCurrentDate();
    const std::string &current_time = gdcm::Util::GetCurrentTime();
    std::string s;
    Plm_image_header plh;

    plh.set_from_plm_image (pli);

    printf ("Hello from gdcm_dose_save: fn = %s\n", dose_fn);

    make_directory_recursive (dose_fn);

    
    /* ----------------------------------------------------------------- */
    /*     Part 1  -- General header                                     */
    /* ----------------------------------------------------------------- */
    /* InstanceCreationDate */
    gf->InsertValEntry (current_date, 0x0008, 0x0012);
    /* InstanceCreationTime */
    gf->InsertValEntry (current_time, 0x0008, 0x0013);
    /* InstanceCreatorUID */
    gf->InsertValEntry (PLM_UID_PREFIX, 0x0008, 0x0014);
    /* SOPClassUID = RTDoseStorage */
    gf->InsertValEntry ("1.2.840.10008.5.1.4.1.1.481.2", 0x0008, 0x0016);
    /* SOPInstanceUID */
    gf->InsertValEntry (gdcm::Util::CreateUniqueUID (PLM_UID_PREFIX), 
			0x0008, 0x0018);
    /* StudyDate */
    gf->InsertValEntry ("20000101", 0x0008, 0x0020);
    /* StudyTime */
    gf->InsertValEntry ("120000", 0x0008, 0x0030);
    /* AccessionNumber */
    gf->InsertValEntry ("", 0x0008, 0x0050);
    /* Modality */
    gf->InsertValEntry ("RTDOSE", 0x0008, 0x0060);
    /* Manufacturer */
    gf->InsertValEntry ("MGH", 0x0008, 0x0070);
    /* ReferringPhysiciansName */
    gf->InsertValEntry ("", 0x0008, 0x0090);
    /* ManufacturersModelName */
    gf->InsertValEntry ("Plastimatch", 0x0008, 0x1090);

#if defined (commentout)
    /* PatientsName */
    if (structures->patient_name) {
	gf->InsertValEntry ((const char*) structures->patient_name->data, 
			    0x0010, 0x0010);
    } else {
	gf->InsertValEntry ("", 0x0010, 0x0010);
    }
    /* PatientID */
    if (structures->patient_id) {
	gf->InsertValEntry ((const char*) structures->patient_id->data, 
			    0x0010, 0x0020);
    } else {
	gf->InsertValEntry ("", 0x0010, 0x0020);
    }
#endif
    /* PatientsName */
    gf->InsertValEntry ("", 0x0010, 0x0010);
    /* PatientID */
    gf->InsertValEntry ("", 0x0010, 0x0020);

    /* PatientsBirthDate */
    gf->InsertValEntry ("", 0x0010, 0x0030);

#if defined (commentout)
    /* PatientsSex */
    if (structures->patient_sex) {
	gf->InsertValEntry ((const char*) structures->patient_sex->data, 
			    0x0010, 0x0040);
    } else {
	gf->InsertValEntry ("", 0x0010, 0x0040);
    }
#endif
    /* PatientsSex */
    gf->InsertValEntry ("", 0x0010, 0x0040);

    /* SliceThickness */
    gf->InsertValEntry ("", 0x0018, 0x0050);
    /* SoftwareVersions */
    gf->InsertValEntry (PLASTIMATCH_VERSION_STRING, 0x0018, 0x1020);

#if defined (commentout)
    /* StudyInstanceUID */
    if (structures->ct_study_uid) {
	gf->InsertValEntry ((const char*) structures->ct_study_uid->data, 
			    0x0020, 0x000d);
    } else {
	gf->InsertValEntry ("", 0x0020, 0x000d);
    }
#endif
    /* StudyInstanceUID */
    gf->InsertValEntry ("", 0x0020, 0x000d);


    /* SeriesInstanceUID */
    gf->InsertValEntry (gdcm::Util::CreateUniqueUID (PLM_UID_PREFIX), 
			0x0020, 0x000e);

#if defined (commentout)
    /* StudyID */
    if (structures->study_id) {
	gf->InsertValEntry ((const char*) structures->study_id->data, 
			    0x0020, 0x0010);
    } else {
	gf->InsertValEntry ("", 0x0020, 0x0010);
    }
#endif
    /* StudyID */
    gf->InsertValEntry ("", 0x0020, 0x0010);

    /* SeriesNumber */
    gf->InsertValEntry ("", 0x0020, 0x0011);
    /* InstanceNumber */
    gf->InsertValEntry ("1", 0x0020, 0x0013);
    /* ImagePositionPatient */
    s = gdcm::Util::Format ("%g\\%g\\%g", plh.m_origin[0], 
	plh.m_origin[1], plh.m_origin[2]);
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

    /* GCS FIX */
#if defined (commentout)
    /* FrameOfReferenceUID */
    gf->InsertValEntry ("XXX.XXX.XXXXXXX", 0x0020, 0x0052);
#endif

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
    s = gdcm::Util::Format ("%g\\%g", plh.m_spacing[0], plh.m_spacing[1]);
    gf->InsertValEntry (s, 0x0028, 0x0030);

    /* BitsAllocated */
    gf->InsertValEntry ("16", 0x0028, 0x0100);
    /* BitsStored */
    gf->InsertValEntry ("16", 0x0028, 0x0101);
    /* HighBit */
    gf->InsertValEntry ("15", 0x0028, 0x0102);
    /* PixelRepresentation */
    gf->InsertValEntry ("0", 0x0028, 0x0103);

    /* Do I need SmallestImagePixelValue, LargestImagePixelValue? */

    /* DoseUnits */
    gf->InsertValEntry ("GY", 0x3004, 0x0002);
    /* DoseType */
    gf->InsertValEntry ("PHYSICAL", 0x3004, 0x0004);
    /* DoseSummationType */
    gf->InsertValEntry ("PLAN", 0x3004, 0x000a);

    /* GridFrameOffsetVector */
    s = gdcm::Util::Format ("%g", plh.m_origin[2]);
    for (i = 1; i < plh.Size(2); i++) {
	s += gdcm::Util::Format ("\\%g", 
	    plh.m_origin[2] + i * plh.m_spacing[2]);
    }
    gf->InsertValEntry (s, 0x3004, 0x000c);

    /* GCS FIX:
       Leave ReferencedRTPlanSequence empty (until I can cross reference) */

    /* We need to convert image to uint16_t, but first we need to 
       scale it.  I'll use the value from XiO: 0.004 */
    float dose_scale = 0.004;
    Plm_image *tmp = pli->clone ();
    tmp->convert (PLM_IMG_TYPE_GPUIT_FLOAT);
    Volume *vol = (Volume*) tmp->m_gpuit;
    volume_scale (vol, 1 / dose_scale);
    tmp->convert (PLM_IMG_TYPE_GPUIT_UINT16);
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
}
