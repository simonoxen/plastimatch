/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plmbase.h"
#include "plmutil.h"
#include "plmsys.h"

#include "plm_math.h"
#include "pstring.h"

void
cxt_load (
    Rtss *rtss,                    /* Output: load into this object */
    Slice_index *rdd,     /* Output: Also set some values here */
    const char *cxt_fn             /* Input: file to load from */
)
{
    FILE* fp;
    Rtss_polyline* curr_contour;
    Rtss_polyline_set *cxt = rtss->m_cxt;

    float val_x = 0;
    float val_y = 0;
    float val_z = 0;

    int have_offset = 0;
    int have_dim = 0;
    int have_spacing = 0;
    float x = 0;
    float y = 0;
    float z = 0;

    fp = fopen (cxt_fn, "r");

    if (!fp) {
	fprintf (stderr, "Could not open contour file for read: %s\n", cxt_fn);
        exit (-1);
    }

    /* Part 1: Dicom info */
    while (1) {
	int tag_idx;
	bstring tag, val;

	tag = bgets ((bNgetc) fgetc, fp, '\n');
        if (!tag) {
            fprintf (stderr, 
                "Error. cxt file is not formatted correctly: %s\n",
                cxt_fn);
            exit (-1);
        }

	btrimws (tag);
        tag_idx = bstrchr (tag, ' ');
	if (tag_idx == BSTR_ERR) {
	    val = 0;
	} else {
	    val = bmidstr (tag, tag_idx, tag->slen);
	    btrimws (val);
	    btrunc (tag, tag_idx);
	}
	//printf ("%s|%s|\n", tag->data, val ? val->data : (unsigned char*) "(null)");

	if (biseqcstr (tag, "ROI_NAMES")) {
	    bdestroy (tag);
	    bdestroy (val);
            break;
        }
        else if (!val) {
	    /* fall through */
	}
        else if (biseqcstr (tag, "PATIENT_NAME")) {
	    rtss->m_meta.set_metadata (0x0010, 0x0010, 
		(const char*) val->data);
	}
        else if (biseqcstr (tag, "PATIENT_ID")) {
	    rtss->m_meta.set_metadata (0x0010, 0x0020, 
		(const char*) val->data);
	}
        else if (biseqcstr (tag, "PATIENT_SEX")) {
	    rtss->m_meta.set_metadata (0x0010, 0x0040, 
		(const char*) val->data);
	}
        else if (biseqcstr (tag, "STUDY_ID")) {
	    if (rdd->m_study_id.empty()) {
		rdd->m_study_id = (const char*) val->data;
	    }
	}
        else if (biseqcstr (tag, "CT_STUDY_UID")) {
	    if (rdd->m_ct_study_uid.empty()) {
		rdd->m_ct_study_uid = (const char*) val->data;
	    }
	}
        else if (biseqcstr (tag, "CT_SERIES_UID")) {
	    if (rdd->m_ct_series_uid.empty()) {
		rdd->m_ct_series_uid = (const char*) val->data;
	    }
	}
        else if (biseqcstr (tag, "CT_FRAME_OF_REFERENCE_UID")) {
	    if (rdd->m_ct_fref_uid.empty()) {
		rdd->m_ct_fref_uid = (const char*) val->data;
	    }
	}
        else if (biseqcstr (tag, "OFFSET")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", 
		    &val_x, &val_y, &val_z)) {
		have_offset = 1;
		cxt->m_offset[0] = val_x;
		cxt->m_offset[1] = val_y;
		cxt->m_offset[2] = val_z;
	    }
	}
        else if (biseqcstr (tag, "DIMENSION")) {
	    int int_x, int_y, int_z;
	    if (3 == sscanf ((const char*) val->data, "%d %d %d", 
                    &int_x, &int_y, &int_z)) {
		have_dim = 1;
		cxt->m_dim[0] = int_x;
		cxt->m_dim[1] = int_y;
		cxt->m_dim[2] = int_z;
	    }
	}
        else if (biseqcstr (tag, "SPACING")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", 
		    &val_x, &val_y, &val_z)) {
		have_spacing = 1;
		cxt->m_spacing[0] = val_x;
		cxt->m_spacing[1] = val_y;
		cxt->m_spacing[2] = val_z;
	    }
	}
	bdestroy (tag);
	bdestroy (val);
    }
    if (have_offset && have_dim && have_spacing) {
	cxt->have_geometry = 1;
    }

    /* Part 2: Structures info */
    printf ("Starting structure parsing\n");
    while (1) {
        char color[CXT_BUFLEN];
        char name[CXT_BUFLEN];
        char buf[CXT_BUFLEN];
	int struct_id;
        char *p;
        int rc;

        p = fgets (buf, CXT_BUFLEN, fp);
        if (!p) {
            fprintf (stderr, 
                "Error. cxt file is not formatted correctly: %s\n",
                cxt_fn);
            exit (-1);
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_id, color, name);
        if (rc != 3) {
            break;
        }
	cxt->add_structure (Pstring (name), Pstring (color), struct_id);
    }

    /* Part 3: Contour info */
    printf ("Starting contour parsing\n");
    while (1) {
	int k;
	int num_pt;
	int struct_id = 0;
	int slice_idx;
	char slice_uid[1024];
	Rtss_structure* curr_structure;

	/* Structure id */
        if (1 != fscanf (fp, " %d", &struct_id)) {
	    /* Normal exit from loop */
	    break;
        }
        fgetc (fp);

        /* Skip contour thickness */
        while (fgetc (fp) != '|') ;

        /* Num vertices: required */
	num_pt = 0;
        if (1 != fscanf (fp, "%d", &num_pt)) {
	    goto not_successful;
        }
        fgetc (fp);

        /* Slice idx: optional */
	slice_idx = -1;
        if (1 != fscanf (fp, "%d", &slice_idx)) {
	    slice_idx = -1;
        }
        fgetc (fp);

        /* Slice uid: optional */
	slice_uid[0] = 0;
        if (1 != fscanf (fp, "%1023[0-9.]", slice_uid)) {
	    slice_uid[0] = 0;
	}
        fgetc (fp);

        curr_structure = cxt->find_structure_by_id (struct_id);

	/* If there is no header line for this structure, we will 
	   skip all contours for the structure. */
	if (!curr_structure) {
	    /* Skip to end of line */
	    while (fgetc (fp) != '\n') ;
	    continue;
	}

        curr_contour = curr_structure->add_polyline ();
        curr_contour->num_vertices = num_pt;
        curr_contour->slice_no = slice_idx;
	if (slice_uid[0]) {
	    curr_contour->ct_slice_uid = slice_uid;
	} else {
	    curr_contour->ct_slice_uid = "";
	}

        curr_contour->x = (float*) malloc (num_pt * sizeof(float));
        curr_contour->y = (float*) malloc (num_pt * sizeof(float));
        curr_contour->z = (float*) malloc (num_pt * sizeof(float));
        if (curr_contour->y == 0 || curr_contour->x == 0) {
            fprintf (stderr, "Error allocating memory");
            exit (-1);
        }
        for (k = 0; k < num_pt; k++) {
	    //long floc;
	    //floc = ftell (fp);
            if (fscanf (fp, "%f\\%f\\%f\\", &x, &y, &z) != 3) {
		goto not_successful;
            }
            curr_contour->x[k] = x;
            curr_contour->y[k] = y;
            curr_contour->z[k] = z;
            x = 0;
            y = 0;
            z = 0;
        }
        slice_idx = 0;
        num_pt = 0;
    }
    fclose (fp);
    return;
not_successful:
    fclose (fp);
    fprintf (stderr, "Error parsing input file: %s\n", cxt_fn);
    exit (1);
}

void
cxt_save (
    Rtss *rtss,                  /* Input: Structure set to save from */
    Slice_index *rdd,   /* Input: Also save some values from here */
    const char* cxt_fn,          /* Input: File to save to */
    bool prune_empty             /* Input: Should we prune empty structures? */
)
{
    FILE *fp;
    Rtss_polyline_set *cxt = rtss->m_cxt;

    /* Prepare output directory */
    make_directory_recursive (cxt_fn);

    fp = fopen (cxt_fn, "wb");
    if (!fp) {
	fprintf (stderr, 
	    "Could not open contour file for write: %s\n", cxt_fn);
        exit (-1);
    }

    /* Part 1: Dicom info */
    if (rdd && rdd->m_ct_series_uid.not_empty()) {
	fprintf (fp, "CT_SERIES_UID %s\n", (const char*) rdd->m_ct_series_uid);
    } else {
	fprintf (fp, "CT_SERIES_UID\n");
    }
    if (rdd && rdd->m_ct_study_uid.not_empty()) {
	fprintf (fp, "CT_STUDY_UID %s\n", (const char*) rdd->m_ct_study_uid);
    } else {
	fprintf (fp, "CT_STUDY_UID\n");
    }
    if (rdd && rdd->m_ct_fref_uid.not_empty()) {
	fprintf (fp, "CT_FRAME_OF_REFERENCE_UID %s\n", 
	    (const char*) rdd->m_ct_fref_uid);
    } else {
	fprintf (fp, "CT_FRAME_OF_REFERENCE_UID\n");
    }
    fprintf (fp, "PATIENT_NAME %s\n",
	rtss->m_meta.get_metadata (0x0010, 0x0010).c_str());
    fprintf (fp, "PATIENT_ID %s\n",
	rtss->m_meta.get_metadata (0x0010, 0x0020).c_str());
    fprintf (fp, "PATIENT_SEX %s\n",
	rtss->m_meta.get_metadata (0x0010, 0x0040).c_str());
    if (rdd && rdd->m_study_id.not_empty()) {
	fprintf (fp, "STUDY_ID %s\n", (const char*) rdd->m_study_id);
    } else {
	fprintf (fp, "STUDY_ID\n");
    }
    if (cxt->have_geometry) {
	fprintf (fp, "OFFSET %g %g %g\n", cxt->m_offset[0],
	    cxt->m_offset[1], cxt->m_offset[2]);
	fprintf (fp, "DIMENSION %u %u %u\n", (unsigned int) cxt->m_dim[0], 
	    (unsigned int) cxt->m_dim[1], (unsigned int) cxt->m_dim[2]);
	fprintf (fp, "SPACING %g %g %g\n", cxt->m_spacing[0], 
	    cxt->m_spacing[1], cxt->m_spacing[2]);
    }

    /* Part 2: Structures info */
    fprintf (fp, "ROI_NAMES\n");
    for (size_t i = 0; i < cxt->num_structures; i++) {
	Rtss_structure *curr_structure = cxt->slist[i];
	if (prune_empty && curr_structure->num_contours <= 0) {
	    continue;
	}
	fprintf (fp, "%d|%s|%s\n", 
	    curr_structure->id, 
	    (curr_structure->color.empty() 
		? "255\\0\\0"
		: (const char*) curr_structure->color), 
	    (const char*) curr_structure->name);
    }
    fprintf (fp, "END_OF_ROI_NAMES\n");

    /* Part 3: Contour info */
    for (size_t i = 0; i < cxt->num_structures; i++) {
	Rtss_structure *curr_structure = cxt->slist[i];
	if (prune_empty && curr_structure->num_contours <= 0) {
	    continue;
	}
	for (size_t j = 0; j < curr_structure->num_contours; j++) {
	    int k;
	    Rtss_polyline *curr_polyline = curr_structure->pslist[j];

	    /* struct_no|contour_thickness|num_points|slice_no|slice_uid|points */
	    /* I don't think contour thickness is used. */
	    fprintf (fp, "%d||%d|", curr_structure->id, 
		curr_polyline->num_vertices);
	    /* slice_no and slice_uid are optional */
	    if (curr_polyline->slice_no >= 0) {
		fprintf (fp, "%d|", curr_polyline->slice_no);
	    } else {
		fprintf (fp, "|");
	    }
	    if (curr_polyline->ct_slice_uid.empty()) {
		fprintf (fp, "|");
	    } else {
		fprintf (fp, "%s|", (const char*) curr_polyline->ct_slice_uid);
	    }
	    for (k = 0; k < curr_polyline->num_vertices; k++) {
		if (k > 0) {
		    fprintf (fp, "\\");
		}
		fprintf (fp, "%f\\%f\\%f",
		    curr_polyline->x[k], 
		    curr_polyline->y[k], 
		    curr_polyline->z[k]);
	    }
	    fprintf (fp, "\n");
	}
    }

    fclose (fp);
}
