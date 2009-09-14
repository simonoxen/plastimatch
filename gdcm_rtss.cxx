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
#include "gdcm_rtss.h"
#include "gdcm_series.h"
#include "print_and_exit.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "readcxt.h"

plastimatch1_EXPORT
void
gdcm_rtss_load (Cxt_structure_list *structures, char *rtss_fn, char *dicom_dir)
{
    gdcm::File *gdcm_file = new gdcm::File;
    gdcm::SeqEntry *seq;
    gdcm::SQItem *item;
    Gdcm_series gs;
    std::string tmp;

    gdcm_file->SetMaxSizeLoadEntry (0xffff);
    gdcm_file->SetFileName (rtss_fn);
    gdcm_file->SetLoadMode (0);
    gdcm_file->Load();

    /* Modality -- better be RTSTRUCT */
    tmp = gdcm_file->GetEntryValue (0x0008, 0x0060);
    if (strncmp (tmp.c_str(), "RTSTRUCT", strlen("RTSTRUCT"))) {
	print_and_exit ("Error.  Input file not an RT structure set: %s\n",
			rtss_fn);
    }

    /* Got the RT struct.  Try to load the corresponding CT. */
    if (dicom_dir) {
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

    /* PatientName */
    tmp = gdcm_file->GetEntryValue (0x0010, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_name = bfromcstr (tmp.c_str());
    }

    /* PatientID */
    tmp = gdcm_file->GetEntryValue (0x0010, 0x0020);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_id = bfromcstr (tmp.c_str());
    }

    /* PatientSex */
    tmp = gdcm_file->GetEntryValue (0x0010, 0x0040);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_sex = bfromcstr (tmp.c_str());
    }

    /* StudyID */
    tmp = gdcm_file->GetEntryValue (0x0020, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->study_id = bfromcstr (tmp.c_str());
    }

    /* ReferencedFramOfReferenceSequence */
    gdcm::SeqEntry *referencedFrameOfReferenceSequence = gdcm_file->GetSeqEntry(0x3006,0x0010);
    item = referencedFrameOfReferenceSequence->GetFirstSQItem();
    /* FrameOfReferenceUID */
    tmp = item->GetEntryValue(0x0020,0x0052);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->ct_series_uid = bfromcstr (tmp.c_str());
    }

    /* StructureSetROISequence */
    seq = gdcm_file->GetSeqEntry (0x3006,0x0020);
    for (item = seq->GetFirstSQItem (); item; item = seq->GetNextSQItem ()) {
	int structure_id;
	std::string roi_number, roi_name;
	roi_number = item->GetEntryValue (0x3006,0x0022);
	roi_name = item->GetEntryValue (0x3006,0x0026);
	if (1 != sscanf (roi_number.c_str(), "%d", &structure_id)) {
	    continue;
	}
	cxt_add_structure (structures, roi_name.c_str(), structure_id);
    }

    /* ROIContourSequence */
    seq = gdcm_file->GetSeqEntry (0x3006,0x0039);
    for (item = seq->GetFirstSQItem (); item; item = seq->GetNextSQItem ()) {
	int structure_id;
	std::string roi_display_color, referenced_roi_number;
	gdcm::SeqEntry *c_seq;
	gdcm::SQItem *c_item;
	Cxt_structure *curr_structure;

	/* Get id and color */
	referenced_roi_number = item->GetEntryValue (0x3006,0x0084);
	roi_display_color = item->GetEntryValue (0x3006,0x002a);
	printf ("RRN = [%s], RDC = [%s]\n", referenced_roi_number.c_str(), roi_display_color.c_str());

	if (1 != sscanf (referenced_roi_number.c_str(), "%d", &structure_id)) {
	    printf ("Error parsing rrn...\n");
	    continue;
	}

	/* Look up the cxt structure for this id */
	curr_structure = cxt_find_structure_by_id (structures, structure_id);
	if (!curr_structure) {
	    printf ("Couldn't reference structure with id %d\n", structure_id);
	    exit (-1);
	}

	/* ContourSequence */
	printf ("Parsing contour_sequence...\n");
	c_seq = item->GetSeqEntry (0x3006,0x0040);
	for (c_item = c_seq->GetFirstSQItem (); c_item; c_item = c_seq->GetNextSQItem ()) {
	    int i, p, n, contour_data_len;
	    int num_points;
	    std::string contour_geometric_type;
	    std::string contour_data;
	    std::string number_of_contour_points;
	    Cxt_polyline *curr_polyline;

	    /* Grab data from dicom */
	    contour_geometric_type = c_item->GetEntryValue (0x3006,0x0042);
	    if (strncmp (contour_geometric_type.c_str(), "CLOSED_PLANAR", strlen("CLOSED_PLANAR"))) {
		/* Might be "POINT".  Do I want to preserve this? */
		printf ("Skipping geometric type: [%s]\n", contour_geometric_type.c_str());
		continue;
	    }
	    number_of_contour_points = c_item->GetEntryValue (0x3006,0x0046);
	    if (1 != sscanf (number_of_contour_points.c_str(), "%d", &num_points)) {
		printf ("Error parsing number_of_contour_points...\n");
		continue;
	    }
	    contour_data = c_item->GetEntryValue (0x3006,0x0050);
	    if (contour_data == gdcm::GDCM_UNFOUND) {
		printf ("Error grabbing contour data.\n");
		continue;
	    }

	    /* Create a new polyline for this structure */
	    curr_polyline = cxt_add_polyline (curr_structure);
	    curr_polyline->slice_no = -1;
	    curr_polyline->num_vertices = num_points;
	    curr_polyline->x = (float*) malloc (num_points * sizeof(float));
	    curr_polyline->y = (float*) malloc (num_points * sizeof(float));
	    curr_polyline->z = (float*) malloc (num_points * sizeof(float));

	    /* Parse dicom data string */
	    i = 0;
	    n = 0;
	    contour_data_len = strlen (contour_data.c_str());
	    for (p = 0; p < 3 * num_points; p++) {
		float f;
		int this_n;
		
		/* Skip \\ */
		if (n < contour_data_len) {
		    if (contour_data.c_str()[n] == '\\') {
			n++;
		    }
		}

		/* Parse float value */
		if (1 != sscanf (&contour_data[n], "%f%n", &f, &this_n)) {
		    printf ("Error parsing data...\n");
		    break;
		}
		n += this_n;

		/* Put value into polyline */
		switch (i) {
		case 0:
		    curr_polyline->x[p/3] = f;
		    break;
		case 1:
		    curr_polyline->y[p/3] = f;
		    break;
		case 2:
		    curr_polyline->z[p/3] = f;
		    break;
		}
		i = (i + 1) % 3;
	    }
	}
    }
}

plastimatch1_EXPORT
void
gdcm_rtss_save (Cxt_structure_list *structures, char *rtss_fn)
{
    gdcm::File *gf = new gdcm::File ();
#if defined (commentout)
    gdcm::FileHelper *gfh = new gdcm::FileHelper (gf);
#endif
    const std::string &current_date = gdcm::Util::GetCurrentDate();
    const std::string &current_time = gdcm::Util::GetCurrentTime();

    /* Due to a bug in gdcm, it is not possible to create a gdcmFile 
	which does not have a (7fe0,0000) PixelDataGroupLength element.
	Therefore we have to write using Document::WriteContent() */
    std::ofstream *fp;
    fp = new std::ofstream (rtss_fn, std::ios::out | std::ios::binary);
    if (*fp == NULL) {
	fprintf (stderr, "Error opening file for write: %s\n", rtss_fn);
	return;
    }
    
    /* TransferSyntaxUID */
    //    gf->InsertValEntry ("ISO_IR 100", 0x0002, 0x0010);
    /* InstanceCreationDate */
    gf->InsertValEntry (current_date, 0x0008, 0x0012);
    /* InstanceCreationTime */
    gf->InsertValEntry (current_time, 0x0008, 0x0013);
    /* InstanceCreatorUID */
    gf->InsertValEntry (PLM_UID_PREFIX, 0x0008, 0x0014);
    /* SOPClassUID = RTStructureSetStorage */
    gf->InsertValEntry ("1.2.840.10008.5.1.4.1.1.481.3", 0x0008, 0x0016);
    /* SOPInstanceUID */
    gf->InsertValEntry (gdcm::Util::CreateUniqueUID (PLM_UID_PREFIX), 0x0008, 0x0018);
    /* StudyDate */
    gf->InsertValEntry ("", 0x0008, 0x0020);
    /* StudyTime */
    gf->InsertValEntry ("", 0x0008, 0x0030);
    /* Modality */
    gf->InsertValEntry ("RTSTRUCT", 0x0008, 0x0060);
    /* AccessionNumber */
    gf->InsertValEntry ("", 0x0008, 0x0050);
    /* Manufacturer */
    gf->InsertValEntry ("Plastimatch", 0x0008, 0x0070);
    /* ReferringPhysiciansName */
    gf->InsertValEntry ("", 0x0008, 0x0090);
    /* StationName */
    gf->InsertValEntry ("", 0x0008, 0x1010);
    /* SeriesDescription */
    gf->InsertValEntry ("Plastimatch structure set", 0x0008, 0x103e);
    /* ManufacturersModelName */
    gf->InsertValEntry ("Plastimatch", 0x0008, 0x1090);
    /* PatientsName */
    gf->InsertValEntry ("", 0x0010, 0x0010);
    /* PatientID */
    gf->InsertValEntry ("", 0x0010, 0x0020);
    /* PatientsBirthDate */
    gf->InsertValEntry ("", 0x0010, 0x0030);
    /* PatientsSex */
    gf->InsertValEntry ("", 0x0010, 0x0040);
    /* SoftwareVersions */
    gf->InsertValEntry (PLASTIMATCH_VERSION_STRING, 0x0018, 0x1020);
    /* PatientPosition */
    // gf->InsertValEntry (xxx, 0x0018, 0x5100);
    /* StudyInstanceUID */
    gf->InsertValEntry ("", 0x0020, 0x000d);
    /* SeriesInstanceUID */
    gf->InsertValEntry ("", 0x0020, 0x000e);
    /* StudyID */
    gf->InsertValEntry ("", 0x0020, 0x0010);
    /* SeriesNumber */
    gf->InsertValEntry ("103", 0x0020, 0x0011);
    /* InstanceNumber */
    gf->InsertValEntry ("1", 0x0020, 0x0013);
    /* StructureSetLabel */
    gf->InsertValEntry ("", 0x3006, 0x0002);
    /* StructureSetName */
    gf->InsertValEntry ("", 0x3006, 0x0004);
    /* StructureSetDate */
    gf->InsertValEntry (current_date, 0x3006, 0x0008);
    /* StructureSetTime */
    gf->InsertValEntry (current_time, 0x3006, 0x0009);

    /* Sequence of CT slices */
    gdcm::SeqEntry *seq;
    gdcm::SQItem *item;
    seq = gf->InsertSeqEntry (0x3006, 0x0010);
    item = new gdcm::SQItem (seq->GetDepthLevel());
    seq->AddSQItem (item, 1);
#if defined (NEED_DICOM_UIDS____)
    for (i = 0; i < foo; i++) {
	item->InsertValEntry (current_time, 0x3006, 0x0009);
    }
#endif

#if defined (commentout)
    gfh->SetWriteTypeToDcmExplVR ();
    gfh->Write (rtss_fn);
#endif

    gf->WriteContent (fp, gdcm::ExplicitVR);
    fp->close();
    delete fp;
}
