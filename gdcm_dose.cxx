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
extern "C"
gpuit_EXPORT
char* file_util_dirname (const char *filename);

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
    gdcm::SeqEntry *seq;
    gdcm::SQItem *item;
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
    int i, j, k;
    gdcm::File *gf = new gdcm::File ();
    Gdcm_series gs;
    const std::string &current_date = gdcm::Util::GetCurrentDate();
    const std::string &current_time = gdcm::Util::GetCurrentTime();

    printf ("Hello from gdcm_dose_save\n");

#if defined (commentout)
    /* Got the RT struct.  Try to load the corresponding CT. */
    if (dicom_dir[0] != '\0') {
	gs.load (dicom_dir);
	gs.get_best_ct ();
	if (gs.m_have_ct) {
	    int d;
	    structures->have_geometry = 1;
	    for (d = 0; d < 3; d++) {
		structures->offset[d] = gs.m_origin[d];
		structures->dim[d] = gs.m_dim[d];
		structures->spacing[d] = gs.m_spacing[d];
	    }
	}
    }


    /* Due to a bug in gdcm, it is not possible to create a gdcmFile 
       which does not have a (7fe0,0000) PixelDataGroupLength element.
       Therefore we have to write using Document::WriteContent() */
    std::ofstream *fp;
    fp = new std::ofstream (dose_fn, std::ios::out | std::ios::binary);
    if (*fp == NULL) {
	fprintf (stderr, "Error opening file for write: %s\n", dose_fn);
	return;
    }
    
    /* ----------------------------------------------------------------- */
    /*     Part 1  -- General header                                     */
    /* ----------------------------------------------------------------- */

    /* From Chang-Yu Wang: 
       Some dicom validation toolkit (such as DVTK dicom editor)
       required the TransferSyntaxUID tag, and commenting out 
       gf->InsertValEntry ("ISO_IR 100", 0x0002, 0x0010); in gdcm_dose.cxx 
       will cause failure to read in. */

    /* TransferSyntaxUID */
    gf->InsertValEntry ("ISO_IR 100", 0x0002, 0x0010);
    /* InstanceCreationDate */
    gf->InsertValEntry (current_date, 0x0008, 0x0012);
    /* InstanceCreationTime */
    gf->InsertValEntry (current_time, 0x0008, 0x0013);
    /* InstanceCreatorUID */
    gf->InsertValEntry (PLM_UID_PREFIX, 0x0008, 0x0014);
    /* SOPClassUID = RTStructureSetStorage */
    gf->InsertValEntry ("1.2.840.10008.5.1.4.1.1.481.3", 0x0008, 0x0016);
    /* SOPInstanceUID */
    gf->InsertValEntry (gdcm::Util::CreateUniqueUID (PLM_UID_PREFIX), 
			0x0008, 0x0018);
    /* StudyDate */
    gf->InsertValEntry ("20000101", 0x0008, 0x0020);
    /* SeriesDate */
    gf->InsertValEntry ("20000101", 0x0008, 0x0021);
    /* AcquisitionDate */
    gf->InsertValEntry ("20000101", 0x0008, 0x0022);
    /* ContentDate */
    gf->InsertValEntry ("20000101", 0x0008, 0x0023);
    /* StudyTime */
    gf->InsertValEntry ("120000", 0x0008, 0x0030);
    /* AccessionNumber */
    gf->InsertValEntry ("", 0x0008, 0x0050);
    /* Modality */
    gf->InsertValEntry ("RTSTRUCT", 0x0008, 0x0060);
    /* Manufacturer */
    gf->InsertValEntry ("MGH", 0x0008, 0x0070);
    /* ReferringPhysiciansName */
    gf->InsertValEntry ("", 0x0008, 0x0090);
    /* ReferringPhysiciansAddress */
    gf->InsertValEntry ("", 0x0008, 0x0092);
    /* ReferringPhysiciansTelephoneNumbers */
    gf->InsertValEntry ("", 0x0008, 0x0094);
    /* StationName */
    gf->InsertValEntry ("", 0x0008, 0x1010);
    /* SeriesDescription */
    gf->InsertValEntry ("Plastimatch structure set", 0x0008, 0x103e);
    /* ManufacturersModelName */
    gf->InsertValEntry ("Plastimatch", 0x0008, 0x1090);
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
    /* PatientsBirthDate */
    gf->InsertValEntry ("", 0x0010, 0x0030);
    /* PatientsSex */
    if (structures->patient_sex) {
	gf->InsertValEntry ((const char*) structures->patient_sex->data, 
			    0x0010, 0x0040);
    } else {
	gf->InsertValEntry ("", 0x0010, 0x0040);
    }
    /* SoftwareVersions */
    gf->InsertValEntry (PLASTIMATCH_VERSION_STRING, 0x0018, 0x1020);
    /* PatientPosition */
    // gf->InsertValEntry (xxx, 0x0018, 0x5100);
    /* StudyInstanceUID */
    if (structures->ct_study_uid) {
	gf->InsertValEntry ((const char*) structures->ct_study_uid->data, 
			    0x0020, 0x000d);
    } else {
	gf->InsertValEntry ("", 0x0020, 0x000d);
    }
    /* SeriesInstanceUID */
    gf->InsertValEntry (gdcm::Util::CreateUniqueUID (PLM_UID_PREFIX), 
			0x0020, 0x000e);
    /* StudyID */
    if (structures->study_id) {
	gf->InsertValEntry ((const char*) structures->study_id->data, 
			    0x0020, 0x0010);
    } else {
	gf->InsertValEntry ("", 0x0020, 0x0010);
    }
    /* SeriesNumber */
    gf->InsertValEntry ("103", 0x0020, 0x0011);
    /* InstanceNumber */
    gf->InsertValEntry ("1", 0x0020, 0x0013);
    /* StructureSetLabel */
    gf->InsertValEntry ("AutoSS", 0x3006, 0x0002);
    /* StructureSetName */
    gf->InsertValEntry ("AutoSS", 0x3006, 0x0004);
    /* StructureSetDate */
    gf->InsertValEntry (current_date, 0x3006, 0x0008);
    /* StructureSetTime */
    gf->InsertValEntry (current_time, 0x3006, 0x0009);

    /* ----------------------------------------------------------------- */
    /*     Part 2  -- UID's for CT series                                */
    /* ----------------------------------------------------------------- */

    /* ReferencedFrameOfReferenceSequence */
    gdcm::SeqEntry *rfor_seq = gf->InsertSeqEntry (0x3006, 0x0010);
    gdcm::SQItem *rfor_item = new gdcm::SQItem (rfor_seq->GetDepthLevel());
    rfor_seq->AddSQItem (rfor_item, 1);
    /* FrameOfReferenceUID */
    if (structures->ct_fref_uid) {
	rfor_item->InsertValEntry ((const char*) 
				   structures->ct_fref_uid->data, 
				   0x0020, 0x0052);
    } else {
	rfor_item->InsertValEntry ("", 0x0020, 0x0052);
    }
    /* RTReferencedStudySequence */
    gdcm::SeqEntry *rtrstudy_seq = rfor_item->InsertSeqEntry (0x3006, 0x0012);
    gdcm::SQItem *rtrstudy_item 
	    = new gdcm::SQItem (rtrstudy_seq->GetDepthLevel());
    rtrstudy_seq->AddSQItem (rtrstudy_item, 1);
    /* ReferencedSOPClassUID = DetachedStudyManagementSOPClass */
    rtrstudy_item->InsertValEntry ("1.2.840.10008.3.1.2.3.1", 0x0008, 0x1150);
    /* ReferencedSOPInstanceUID */
    if (structures->ct_study_uid) {
	rtrstudy_item->InsertValEntry ((const char*) 
				       structures->ct_study_uid->data, 
				       0x0008, 0x1155);
    } else {
	rtrstudy_item->InsertValEntry ("", 0x0008, 0x1155);
    }
    /* RTReferencedSeriesSequence */
    gdcm::SeqEntry *rtrseries_seq 
	    = rtrstudy_item->InsertSeqEntry (0x3006, 0x0014);
    gdcm::SQItem *rtrseries_item 
	    = new gdcm::SQItem (rtrseries_seq->GetDepthLevel());
    rtrseries_seq->AddSQItem (rtrseries_item, 1);
    /* SeriesInstanceUID */
    if (structures->ct_series_uid) {
	rtrseries_item->InsertValEntry ((const char*) 
					structures->ct_series_uid->data, 
					0x0020, 0x000e);
    } else {
	rtrseries_item->InsertValEntry ("", 0x0020, 0x000e);
    }
    /* ContourImageSequence */
    gdcm::SeqEntry *ci_seq = rtrseries_item->InsertSeqEntry (0x3006, 0x0016);
    if (gs.m_have_ct) {
	int i = 1;
	gdcm::FileList *file_list = gs.m_ct_file_list;
	for (gdcm::FileList::iterator it =  file_list->begin();
	     it != file_list->end(); 
	     ++it)
	{
	    /* Get SOPInstanceUID of CT */
	    std::string tmp = (*it)->GetEntryValue (0x0008, 0x0018);
	    /* Put item into sequence */
	    gdcm::SQItem *ci_item = new gdcm::SQItem (ci_seq->GetDepthLevel());
	    ci_seq->AddSQItem (ci_item, i++);
	    /* ReferencedSOPClassUID = CTImageStorage */
	    ci_item->InsertValEntry ("1.2.840.10008.5.1.4.1.1.2", 
				     0x0008, 0x1150);
	    /* Put ReferencedSOPInstanceUID */
	    ci_item->InsertValEntry (tmp, 0x0008, 0x1155);
	}
    }
    else {
	/* What to do here? */
	printf ("Warning: CT not found. "
		"ContourImageSequence not generated.\n");
    }

    /* ----------------------------------------------------------------- */
    /*     Part 3  -- Structure info                                     */
    /* ----------------------------------------------------------------- */

    /* StructureSetROISequence */
    gdcm::SeqEntry *ssroi_seq = gf->InsertSeqEntry (0x3006, 0x0020);
    for (i = 0; i < structures->num_structures; i++) {
	gdcm::SQItem *ssroi_item 
		= new gdcm::SQItem (ssroi_seq->GetDepthLevel());
	ssroi_seq->AddSQItem (ssroi_item, i+1);
	/* ROINumber */
	ssroi_item->InsertValEntry (gdcm::Util::Format 
				    ("%d", structures->slist[i].id),
				    0x3006, 0x0022);
	/* ReferencedFrameOfReferenceUID */
	if (structures->ct_fref_uid) {
	    ssroi_item->InsertValEntry ((const char*) 
					structures->ct_fref_uid->data, 
					0x3006, 0x0024);
	} else {
	    ssroi_item->InsertValEntry ("", 0x3006, 0x0024);
	}
	/* ROIName */
	ssroi_item->InsertValEntry (structures->slist[i].name, 0x3006, 0x0026);
	/* ROIGenerationAlgorithm */
	ssroi_item->InsertValEntry ("", 0x3006, 0x0036);
    }

    /* ----------------------------------------------------------------- */
    /*     Part 4  -- Contour info                                       */
    /* ----------------------------------------------------------------- */

    /* ROIContourSequence */
    gdcm::SeqEntry *roic_seq = gf->InsertSeqEntry (0x3006, 0x0039);
    for (i = 0; i < structures->num_structures; i++) {
	Cxt_structure *curr_structure = &structures->slist[i];
	gdcm::SQItem *roic_item 
		= new gdcm::SQItem (roic_seq->GetDepthLevel());
	roic_seq->AddSQItem (roic_item, i+1);
	
	/* ROIDisplayColor */
	if (curr_structure->color) {
	    roic_item->InsertValEntry ((const char*) 
				       curr_structure->color->data,
				       0x3006, 0x002a);
	} else {
	    roic_item->InsertValEntry ("255\\0\\0", 0x3006, 0x002a);
	}
	/* ContourSequence */
	gdcm::SeqEntry *c_seq = roic_item->InsertSeqEntry (0x3006, 0x0040);
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Cxt_polyline *curr_contour = &curr_structure->pslist[j];
	    if (curr_contour->num_vertices <= 0) continue;

	    /* GE -> XiO transfer does not work if contour does not have 
	       corresponding slice uid */
	    if (! curr_contour->ct_slice_uid) continue;

	    gdcm::SQItem *c_item = new gdcm::SQItem (c_seq->GetDepthLevel());
	    c_seq->AddSQItem (c_item, j+1);
	    /* ContourImageSequence */
	    if (curr_contour->ct_slice_uid) {
		gdcm::SeqEntry *ci_seq 
			= c_item->InsertSeqEntry (0x3006, 0x0016);
		gdcm::SQItem *ci_item 
			= new gdcm::SQItem (ci_seq->GetDepthLevel());
		ci_seq->AddSQItem (ci_item, 1);
		/* ReferencedSOPClassUID = CTImageStorage */
		ci_item->InsertValEntry ("1.2.840.10008.5.1.4.1.1.2", 
					 0x0008, 0x1150);
		/* ReferencedSOPInstanceUID */
		ci_item->InsertValEntry ((const char*) 
					 curr_contour->ct_slice_uid->data, 
					 0x0008, 0x1155);
	    }
	    /* ContourGeometricType */
	    c_item->InsertValEntry ("CLOSED_PLANAR", 0x3006, 0x0042);
	    /* NumberOfContourPoints */
	    c_item->InsertValEntry (gdcm::Util::Format 
				     ("%d", curr_contour->num_vertices),
				     0x3006, 0x0046);
	    /* ContourData */
	    std::string contour_string 
		    = gdcm::Util::Format ("%g\\%g\\%g",
					  curr_contour->x[0],
					  curr_contour->y[0],
					  curr_contour->z[0]);
	    for (k = 1; k < curr_contour->num_vertices; k++) {
		contour_string += gdcm::Util::Format ("\\%g\\%g\\%g",
						      curr_contour->x[k],
						      curr_contour->y[k],
						      curr_contour->z[k]);
	    }
	    c_item->InsertValEntry (contour_string, 0x3006, 0x0050);
	}
	/* ReferencedROINumber */
	roic_item->InsertValEntry (gdcm::Util::Format 
				   ("%d", curr_structure->id),
				   0x3006, 0x0084);
    }

    /* ----------------------------------------------------------------- */
    /*     Part 5  -- More structure info                                */
    /* ----------------------------------------------------------------- */

    /* RTROIObservationsSequence */
    gdcm::SeqEntry *rtroio_seq = gf->InsertSeqEntry (0x3006, 0x0080);
    for (i = 0; i < structures->num_structures; i++) {
	Cxt_structure *curr_structure = &structures->slist[i];
	gdcm::SQItem *rtroio_item 
		= new gdcm::SQItem (rtroio_seq->GetDepthLevel());
	rtroio_seq->AddSQItem (rtroio_item, i+1);
	/* ObservationNumber */
	rtroio_item->InsertValEntry (gdcm::Util::Format 
				     ("%d", curr_structure->id),
				     0x3006, 0x0082);
	/* ReferencedROINumber */
	rtroio_item->InsertValEntry (gdcm::Util::Format 
				     ("%d", curr_structure->id),
				     0x3006, 0x0084);
	/* ROIObservationLabel */
	rtroio_item->InsertValEntry (curr_structure->name, 0x3006, 0x0085);
	/* RTROIInterpretedType */
	rtroio_item->InsertValEntry ("", 0x3006, 0x00a4);
	/* ROIInterpreter */
	rtroio_item->InsertValEntry ("", 0x3006, 0x00a6);
    }

    /* Do the actual writing out to file */
    gf->WriteContent (fp, gdcm::ExplicitVR);
    fp->close();
    delete fp;
#endif
}
