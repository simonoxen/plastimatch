/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_series.h"
#include "math_util.h"
#include "print_and_exit.h"
#include "rtds.h"
#include "rtss.h"

void
Dcmtk_series::rtss_load (
    Rtds *rtds                       /* Output: this gets updated */
)
{
    Rtss *rtss = new Rtss (rtds);
    rtds->m_ss_image = rtss;
    Rtss_polyline_set *cxt = rtss->m_cxt;
    
    std::string modality = this->get_modality();
    if (modality == "RTSTRUCT") {
        printf ("Trying to load rt structure set.\n");
    } else {
        print_and_exit ("Oops.\n");
    }

    /* ReferencedFrameOfReferenceSequence */
    DcmSequenceOfItems *seq = 0;
    bool rc = m_flist.front()->get_sequence (
        DCM_ReferencedFrameOfReferenceSequence, seq);
    if (!rc) {
        printf ("Huh?  No sequence?? %p\n", seq);
    } else {
        printf ("Wow... cool!\n");
    }

#if defined (commentout)
    gdcm::SeqEntry *rfor_seq = rtss_file->GetSeqEntry (0x3006,0x0010);
    if (rfor_seq) {

	/* FrameOfReferenceUID */
	item = rfor_seq->GetFirstSQItem ();
	if (item) {
	    tmp = item->GetEntryValue (0x0020,0x0052);
	    if (rdd->m_ct_fref_uid.empty()) {
		if (tmp != gdcm::GDCM_UNFOUND) {
		    rdd->m_ct_fref_uid = tmp.c_str();
		}
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
			    if (rdd->m_ct_series_uid.empty()) {
				if (tmp != gdcm::GDCM_UNFOUND) {
				    rdd->m_ct_series_uid = tmp.c_str();
				}
			    }
			}
		    }
		}
	    }
	}
    }

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
	cxt->add_structure (Pstring (roi_name.c_str()), 
	    Pstring (), structure_id);
    }

    /* ROIContourSequence */
    seq = rtss_file->GetSeqEntry (0x3006,0x0039);
    for (item = seq->GetFirstSQItem (); item; item = seq->GetNextSQItem ()) {
	int structure_id;
	std::string roi_display_color, referenced_roi_number;
	gdcm::SeqEntry *c_seq;
	gdcm::SQItem *c_item;
	Rtss_structure *curr_structure;

	/* Get id and color */
	referenced_roi_number = item->GetEntryValue (0x3006,0x0084);
	roi_display_color = item->GetEntryValue (0x3006,0x002a);
	printf ("RRN = [%s], RDC = [%s]\n", referenced_roi_number.c_str(), roi_display_color.c_str());

	if (1 != sscanf (referenced_roi_number.c_str(), "%d", &structure_id)) {
	    printf ("Error parsing rrn...\n");
	    continue;
	}

	/* Look up the cxt structure for this id */
	curr_structure = cxt->find_structure_by_id (structure_id);
	if (!curr_structure) {
	    printf ("Couldn't reference structure with id %d\n", structure_id);
	    exit (-1);
	}
	curr_structure->set_color (roi_display_color.c_str());

	/* ContourSequence */
	c_seq = item->GetSeqEntry (0x3006,0x0040);
	if (c_seq) {
	    for (c_item = c_seq->GetFirstSQItem (); c_item; c_item = c_seq->GetNextSQItem ()) {
		int i, p, n, contour_data_len;
		int num_points;
		std::string contour_geometric_type;
		std::string contour_data;
		std::string number_of_contour_points;
		Rtss_polyline *curr_polyline;

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
		curr_polyline = curr_structure->add_polyline ();
		curr_polyline->slice_no = -1;
		//curr_polyline->ct_slice_uid = "";
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
    printf ("Loading complete.\n");
    delete rtss_file;
#endif
}
