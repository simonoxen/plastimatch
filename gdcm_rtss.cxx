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
#include "cxt_io.h"

plastimatch1_EXPORT
void
gdcm_rtss_load (Cxt_structure_list *structures, char *rtss_fn, char *dicom_dir)
{
    gdcm::File *rtss_file = new gdcm::File;
    gdcm::SeqEntry *seq;
    gdcm::SQItem *item;
    Gdcm_series gs;
    std::string tmp;

    rtss_file->SetMaxSizeLoadEntry (0xffff);
    rtss_file->SetFileName (rtss_fn);
    rtss_file->SetLoadMode (0);
    rtss_file->Load();


    /* Modality -- better be RTSTRUCT */
    tmp = rtss_file->GetEntryValue (0x0008, 0x0060);
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
    tmp = rtss_file->GetEntryValue (0x0010, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_name = bfromcstr (tmp.c_str());
    }

    /* PatientID */
    tmp = rtss_file->GetEntryValue (0x0010, 0x0020);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_id = bfromcstr (tmp.c_str());
    }

    /* PatientSex */
    tmp = rtss_file->GetEntryValue (0x0010, 0x0040);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->patient_sex = bfromcstr (tmp.c_str());
    }

    /* StudyID */
    tmp = rtss_file->GetEntryValue (0x0020, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	structures->study_id = bfromcstr (tmp.c_str());
    }

    /* If we have a CT series, get the uids from there */
    if (gs.m_have_ct) {
	gdcm::File *ct_file = gs.get_ct_slice ();
	
	/* StudyInstanceUID */
	tmp = ct_file->GetEntryValue (0x0020, 0x000d);
	structures->ct_study_uid = bfromcstr (tmp.c_str());
	
	/* SeriesInstanceUID */
	tmp = ct_file->GetEntryValue (0x0020, 0x000e);
	structures->ct_series_uid = bfromcstr (tmp.c_str());
	
	/* FrameOfReferenceUID */
	tmp = ct_file->GetEntryValue (0x0020, 0x0052);
	structures->ct_fref_uid = bfromcstr (tmp.c_str());
    } 

    /* Otherwise, no CT series, so we get the UIDs from the RT structure set */
    else {

	/* StudyInstanceUID */
	tmp = rtss_file->GetEntryValue (0x0020, 0x000d);
	structures->ct_study_uid = bfromcstr (tmp.c_str());

	/* ReferencedFrameOfReferenceSequence */
	gdcm::SeqEntry *rfor_seq = rtss_file->GetSeqEntry (0x3006,0x0010);
	if (rfor_seq) {

	    /* FrameOfReferenceUID */
	    item = rfor_seq->GetFirstSQItem ();
	    if (item) {
		tmp = item->GetEntryValue (0x0020,0x0052);
		if (tmp != gdcm::GDCM_UNFOUND) {
		    structures->ct_fref_uid = bfromcstr (tmp.c_str());
		}
	
		/* RTReferencedStudySequence */
		gdcm::SeqEntry *rtrstudy_seq 
			= item->GetSeqEntry (0x3006, 0x0012);
		if (rtrstudy_seq) {
	
		    /* RTReferencedSeriesSequence */
		    item = rtrstudy_seq->GetFirstSQItem ();
		    if (item) {
			gdcm::SeqEntry *rtrseries_seq 
				= item->GetSeqEntry (0x3006, 0x0014);
			if (rtrseries_seq) {
			    item = rtrseries_seq->GetFirstSQItem ();

			    /* SeriesInstanceUID */
			    if (item) {
				tmp = item->GetEntryValue (0x0020, 0x000e);
				if (tmp != gdcm::GDCM_UNFOUND) {
				    structures->ct_series_uid 
					    = bfromcstr (tmp.c_str());
				}
			    }
			}
		    }
		}
	    }
	}
    }

    printf ("Finished uid parsing\n");


    /* StructureSetROISequence */
    seq = rtss_file->GetSeqEntry (0x3006,0x0020);
    for (item = seq->GetFirstSQItem (); item; item = seq->GetNextSQItem ()) {
	int structure_id;
	std::string roi_number, roi_name;
	roi_number = item->GetEntryValue (0x3006,0x0022);
	roi_name = item->GetEntryValue (0x3006,0x0026);
	if (1 != sscanf (roi_number.c_str(), "%d", &structure_id)) {
	    continue;
	}
	cxt_add_structure (structures, roi_name.c_str(), 0, structure_id);
    }

    /* ROIContourSequence */
    seq = rtss_file->GetSeqEntry (0x3006,0x0039);
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
	    if (num_points <= 0) {
		/* Polyline with zero points?  Skip it. */
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
	    curr_polyline->ct_slice_uid = 0;
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
	    /* Find matching CT slice at this z location */
	    if (gs.m_have_ct) {
		gs.get_slice_info (&curr_polyline->slice_no,
				   &curr_polyline->ct_slice_uid,
				   curr_polyline->z[0]);
	    }
	}
    }
    printf ("Loading complete.\n");

}

plastimatch1_EXPORT
void
gdcm_rtss_save (Cxt_structure_list *structures, char *rtss_fn, char *dicom_dir)
{
    int i, j, k;
    gdcm::File *gf = new gdcm::File ();
#if defined (commentout)
    gdcm::FileHelper *gfh = new gdcm::FileHelper (gf);
#endif
    Gdcm_series gs;
    const std::string &current_date = gdcm::Util::GetCurrentDate();
    const std::string &current_time = gdcm::Util::GetCurrentTime();

    printf ("Hello from gdcm_rtss_save\n");

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


    /* Due to a bug in gdcm, it is not possible to create a gdcmFile 
       which does not have a (7fe0,0000) PixelDataGroupLength element.
       Therefore we have to write using Document::WriteContent() */
    std::ofstream *fp;
    fp = new std::ofstream (rtss_fn, std::ios::out | std::ios::binary);
    if (*fp == NULL) {
	fprintf (stderr, "Error opening file for write: %s\n", rtss_fn);
	return;
    }
    
    /* ----------------------------------------------------------------- */
    /*     Part 1  -- General header                                     */
    /* ----------------------------------------------------------------- */

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
    gf->InsertValEntry (gdcm::Util::CreateUniqueUID (PLM_UID_PREFIX), 
			0x0008, 0x0018);
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
    gf->InsertValEntry ("", 0x3006, 0x0002);
    /* StructureSetName */
    gf->InsertValEntry ("", 0x3006, 0x0004);
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
	    ci_item->InsertValEntry ("CTImageStorage", 0x0008, 0x1150);
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
		ci_item->InsertValEntry ("CTImageStorage", 0x0008, 0x1150);
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
}
