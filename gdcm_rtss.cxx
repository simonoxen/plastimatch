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
#include "gdcm_rtss.h"
#include "readcxt.h"

plastimatch1_EXPORT
void
gdcm_rtss_load (Cxt_structure_list *structures, char *rtss_fn, char *dicom_dir)
{
    gdcm::File *gdcm_file = new gdcm::File;
    gdcm::SeqEntry *seq;
    gdcm::SQItem *item;

    gdcm_file->SetMaxSizeLoadEntry (0xffff);
    gdcm_file->SetFileName (rtss_fn);
    gdcm_file->SetLoadMode (0);	    // ??
    bool headerLoaded = gdcm_file->Load();

    std::string foo = gdcm_file->GetEntryValue (0x0008, 0x0060);
    printf ("0x0008,0x0060 = %s\n", foo.c_str());

    /* ReferencedFramOfReferenceSequence */
    gdcm::SeqEntry *referencedFrameOfReferenceSequence = gdcm_file->GetSeqEntry(0x3006,0x0010);
    item = referencedFrameOfReferenceSequence->GetFirstSQItem();
    foo = item->GetEntryValue(0x0020,0x0052);
    printf ("0x0020,0x0052 = %s\n", foo.c_str());

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
    gdcm::File *header = new gdcm::File();

    /* Due to a bug in gdcm, it is not possible to create a gdcmFile 
	which does not have a (7fe0,0000) PixelDataGroupLength element.
	Therefore we have to write using Document::WriteContent() */
    std::ofstream *fp;
    fp = new std::ofstream (rtss_fn, std::ios::out | std::ios::binary);
    if (*fp == NULL) {
	fprintf (stderr, "Error opening file for write: %s\n", rtss_fn);
	return;
    }

    //bstd::string value;
    //MetaDataDictionary & dict = this->GetMetaDataDictionary();

    header->InsertValEntry ("FOOBAR", 0x0010, 0x0010);
    //header->Write (rtss_fn, gdcm::ExplicitVR);

    header->WriteContent (fp, gdcm::ExplicitVR);
    fp->close();
    delete fp;
}
