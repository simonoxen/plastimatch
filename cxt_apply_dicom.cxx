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
#include "cxt_apply_dicom.h"

plastimatch1_EXPORT
void
cxt_apply_dicom_dir (Cxt_structure_list *structures, char *dicom_dir)
{
#if defined (commentout)
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
	cxt_add_structure (structures, roi_name.c_str(), structure_id);
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

#endif
}
