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
#include "rtss_structure.h"

void
Dcmtk_series::rtss_load (
    Rtds *rtds                       /* Output: this gets updated */
)
{
    Rtss *rtss = new Rtss (rtds);
    rtds->m_ss_image = rtss;
    Rtss_polyline_set *cxt = new Rtss_polyline_set;
    rtss->m_cxt = cxt;
    
    /* Modality -- better be RTSTRUCT */
    std::string modality = this->get_modality();
    if (modality == "RTSTRUCT") {
        printf ("Trying to load rt structure set.\n");
    } else {
        print_and_exit ("Oops.\n");
    }

    /* FIX: load metadata such as patient name, etc. */

    /* ReferencedFrameOfReferenceSequence */
    DcmSequenceOfItems *seq = 0;
    bool rc = m_flist.front()->get_sequence (
        DCM_ReferencedFrameOfReferenceSequence, seq);
    if (!rc) {
        printf ("Huh? Why no RFOR sequence???\n");
    }
    /* FIX: need to stash the slice UIDs */

    /* StructureSetROISequence */
    seq = 0;
    rc = m_flist.front()->get_sequence (DCM_StructureSetROISequence, seq);
    if (rc) {
        for (unsigned long i = 0; i < seq->card(); i++) {
            int structure_id;
            OFCondition orc;
            const char *val = 0;
            orc = seq->getItem(i)->findAndGetString (DCM_ROINumber, val);
            if (!orc.good()) {
                continue;
            }
            if (1 != sscanf (val, "%d", &structure_id)) {
                continue;
            }
            val = 0;
            orc = seq->getItem(i)->findAndGetString (DCM_ROIName, val);
            printf ("Adding structure (%d), %s\n", structure_id, val);
            cxt->add_structure (Pstring (val), Pstring (), structure_id);
        }
    }

    /* ROIContourSequence */
    seq = 0;
    rc = m_flist.front()->get_sequence (DCM_ROIContourSequence, seq);
    if (rc) {
        for (unsigned long i = 0; i < seq->card(); i++) {
            Rtss_structure *curr_structure;
            int structure_id;
            OFCondition orc;
            const char *val = 0;
            DcmItem *item = seq->getItem(i);

            /* Get ID and color */
            orc = item->findAndGetString (DCM_ReferencedROINumber, val);
            if (!orc.good()) {
                printf ("Error finding DCM_ReferencedROINumber.\n");
                continue;
            }
            if (1 != sscanf (val, "%d", &structure_id)) {
                continue;
            }
            val = 0;
            orc = item->findAndGetString (DCM_ROIDisplayColor, val);
            printf ("Structure %d has color %s\n", structure_id, val);

            /* Look up the structure for this id and set color */
            curr_structure = cxt->find_structure_by_id (structure_id);
            if (!curr_structure) {
                printf ("Couldn't reference structure with id %d\n", 
                    structure_id);
                continue;
            }
            curr_structure->set_color (val);

            /* ContourSequence */
            DcmSequenceOfItems *c_seq = 0;
            orc = item->findAndGetSequence (DCM_ContourSequence, c_seq);
            if (!orc.good()) {
                printf ("Error finding DCM_ContourSequence.\n");
                continue;
            }
            for (unsigned long j = 0; j < c_seq->card(); j++) {
		int i, p, n, contour_data_len;
		int num_points;
		const char *contour_geometric_type;
		const char *contour_data;
		const char *number_of_contour_points;
		Rtss_polyline *curr_polyline;
                DcmItem *c_item = c_seq->getItem(j);

		/* ContourGeometricType */
                orc = c_item->findAndGetString (DCM_ContourGeometricType, 
                    contour_geometric_type);
                if (!orc.good()) {
		    printf ("Error finding DCM_ContourGeometricType.\n");
                    continue;
                }
		if (strncmp (contour_geometric_type, "CLOSED_PLANAR", 
                        strlen("CLOSED_PLANAR"))) {
		    /* Might be "POINT".  Do I want to preserve this? */
		    printf ("Skipping geometric type: [%s]\n", 
                        contour_geometric_type);
		    continue;
		}

                /* NumberOfContourPoints */
                orc = c_item->findAndGetString (DCM_NumberOfContourPoints,
                    number_of_contour_points);
                if (!orc.good()) {
		    printf ("Error finding DCM_NumberOfContourPoints.\n");
                    continue;
                }
		if (1 != sscanf (number_of_contour_points, "%d", &num_points)) {
		    printf ("Error parsing number_of_contour_points...\n");
		    continue;
		}
		if (num_points <= 0) {
		    /* Polyline with zero points?  Skip it. */
		    continue;
		}
                printf ("Contour %d points\n", num_points);

                /* ContourData */
                orc = c_item->findAndGetString (DCM_ContourData, contour_data);
                if (!orc.good()) {
		    printf ("Error finding DCM_ContourData.\n");
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
		contour_data_len = strlen (contour_data);
		for (p = 0; p < 3 * num_points; p++) {
		    float f;
		    int this_n;
		
		    /* Skip \\ */
		    if (n < contour_data_len) {
			if (contour_data[n] == '\\') {
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
    printf ("%p %p %p\n", rtds,
        rtds->m_ss_image, rtds->m_ss_image->m_cxt);

}
