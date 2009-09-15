/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "bstrlib.h"
#include "readcxt.h"

plastimatch1_EXPORT
void
cxt_initialize (Cxt_structure_list* structures)
{
    memset (structures, 0, sizeof (Cxt_structure_list));
}

plastimatch1_EXPORT
void
cxt_add_structure (Cxt_structure_list* structures, const char *structure_name,
		   int structure_id)
{
    Cxt_structure* new_structure;

    structures->num_structures++;
    structures->slist = (Cxt_structure*) 
	    realloc (structures->slist, 
		     structures->num_structures * sizeof(Cxt_structure));
    new_structure = &structures->slist[structures->num_structures - 1];
    memset (new_structure, 0, sizeof(Cxt_structure));
    strncpy (new_structure->name, structure_name, CXT_BUFLEN);
    new_structure->name[CXT_BUFLEN-1] = 0;
    new_structure->id = structure_id;
}

plastimatch1_EXPORT
Cxt_polyline*
cxt_add_polyline (Cxt_structure* structure)
{
    Cxt_polyline* new_polyline;

    structure->num_contours++;
    structure->pslist = (Cxt_polyline*) realloc (structure->pslist,
						structure->num_contours * sizeof(Cxt_polyline));

    new_polyline = &structure->pslist[structure->num_contours - 1];
    memset (new_polyline, 0, sizeof(Cxt_polyline));
    return new_polyline;
}

plastimatch1_EXPORT
Cxt_structure*
cxt_find_structure_by_id (Cxt_structure_list* structures, int structure_id)
{
    int i;
    Cxt_structure* curr_structure;

    for (i = 0; i < structures->num_structures; i++) {
	curr_structure = &structures->slist[i];
	if (curr_structure->id == structure_id) {
	    return curr_structure;
	}
    }
    return 0;
}

plastimatch1_EXPORT
void
cxt_debug_structures (Cxt_structure_list* structures)
{
    int i;
    Cxt_structure* curr_structure;

    for (i = 0; i < structures->num_structures; i++) {
        curr_structure = &structures->slist[i];
	printf ("%d %d %s\n", i, curr_structure->id, curr_structure->name);
    }
}

plastimatch1_EXPORT
void
cxt_read (Cxt_structure_list* structures, const char* cxt_fn)
{
    FILE* fp;
    Cxt_structure* curr_structure = (Cxt_structure*) malloc (sizeof(Cxt_structure));
    Cxt_polyline* curr_contour;

    float val_x = 0;
    float val_y = 0;
    float val_z = 0;

    int struct_no = 0;
    int old_struct_no = -1;
    int contour_no = 0;

    int have_offset = 0;
    int have_dim = 0;
    int have_spacing = 0;
    float x = 0;
    float y = 0;
    float z = 0;

    memset (curr_structure, 0, sizeof(Cxt_structure));
    curr_structure->num_contours = 0;

    fp = fopen (cxt_fn, "r");

    // if (fp) b = bgets ((bNgetc) fgetc, fp, '\n');

    if (!fp) {
	printf ("Could not open contour file for read: %s\n", cxt_fn);
        exit (-1);
    }

    printf ("Loading...\n");
    /* Part 1: Dicom info */
    while (1) {
	int tag_idx;
	bstring tag, val;

	tag = bgets ((bNgetc) fgetc, fp, '\n');
        if (!tag) {
            fprintf (stderr, "ERROR: Your file is not formatted correctly!\n");
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
        else if (biseqcstr (tag, "PATIENT_NAME")) {
	    structures->patient_name = bstrcpy (val);
	}
        else if (biseqcstr (tag, "PATIENT_ID")) {
	    structures->patient_id = bstrcpy (val);
	}
        else if (biseqcstr (tag, "PATIENT_SEX")) {
	    structures->patient_sex = bstrcpy (val);
	}
        else if (biseqcstr (tag, "STUDY_ID")) {
	    structures->study_id = bstrcpy (val);
	}
        else if (biseqcstr (tag, "CT_STUDY_UID")) {
	    structures->ct_study_uid = bstrcpy (val);
	}
        else if (biseqcstr (tag, "CT_SERIES_UID")) {
	    structures->ct_series_uid = bstrcpy (val);
	}
        else if (biseqcstr (tag, "CT_FRAME_OF_REFERENCE_UID")) {
	    structures->ct_fref_uid = bstrcpy (val);
	}
        else if (biseqcstr (tag, "OFFSET")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", &val_x, &val_y, &val_z)) {
		have_offset = 1;
		structures->offset[0] = val_x;
		structures->offset[1] = val_y;
		structures->offset[2] = val_z;
	    }
	}
        else if (biseqcstr (tag, "DIMENSION")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", &val_x, &val_y, &val_z)) {
		have_dim = 1;
		structures->dim[0] = val_x;
		structures->dim[1] = val_y;
		structures->dim[2] = val_z;
	    }
	}
        else if (biseqcstr (tag, "SPACING")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", &val_x, &val_y, &val_z)) {
		have_spacing = 1;
		structures->spacing[0] = val_x;
		structures->spacing[1] = val_y;
		structures->spacing[2] = val_z;
	    }
	}
	bdestroy (tag);
	bdestroy (val);
    }
    if (have_offset && have_dim && have_spacing) {
	structures->have_geometry = 1;
    }

    /* Part 2: Structures info */
    while (1) {
        char color[CXT_BUFLEN];
        char name[CXT_BUFLEN];
        char buf[CXT_BUFLEN];
	int struct_id;
        char *p;
        int rc;

        p = fgets (buf, CXT_BUFLEN, fp);
        if (!p) {
            fprintf (stderr, "ERROR: Your file is not formatted correctly!\n");
            exit (-1);
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_id, color, name);
        if (rc != 3) {
            break;
        }

        structures->num_structures++;
        structures->slist = (Cxt_structure*) 
		realloc (structures->slist,
			 structures->num_structures * sizeof(Cxt_structure));
        curr_structure = &structures->slist[structures->num_structures - 1];
        strcpy (curr_structure->name, name);
        curr_structure->id = struct_id;
        curr_structure->num_contours = 0;
        curr_structure->pslist = 0;
        printf ("Cxt_structure: %s\n", curr_structure->name);
    }

    /* Part 3: Contour info */
    while (1) {
	int k;
	int num_pt;
	int slice_idx;
	char slice_uid[1024];

	/* Structure no */
        if (1 != fscanf (fp, " %d", &struct_no)) {
	    /* Normal exit from loop */
	    break;
        }
        fgetc (fp);

        /* Skip contour thickness */
        while (fgetc (fp) != '|') ;

        /* Num vertices */
	num_pt = 0;
        if (1 != fscanf (fp, "%d", &num_pt)) {
	    //goto not_successful;
        }
        fgetc (fp);

        /* Slice idx */
	slice_idx = -1;
        if (1 != fscanf (fp, "%d", &slice_idx)) {
	    //goto not_successful;
        }
        fgetc (fp);

        /* Slice uid */
	slice_uid[0] = 0;
        if (1 != fscanf (fp, "%1023[0-9.]", slice_uid)) {
	    //goto not_successful;
	}
        fgetc (fp);

        //printf ("%d %d %d %s\n", struct_no, num_pt, slice_idx, slice_uid);

        if (struct_no != old_struct_no) {
            old_struct_no = struct_no;
            contour_no = 0;
        }
        curr_structure = &structures->slist[struct_no - 1];
        //printf ("Gonna realloc %p, %d\n", curr_structure->pslist, contour_no);
        curr_structure->pslist = (Cxt_polyline*) 
		realloc (curr_structure->pslist, 
			 (contour_no + 1) * sizeof(Cxt_polyline));
        //printf ("Gonna dereference pslist\n");
        curr_contour = &curr_structure->pslist[contour_no];
        curr_contour->num_vertices = num_pt;
        curr_contour->slice_no = slice_idx;
	if (slice_uid[0]) {
	    curr_contour->ct_slice_uid = bfromcstr (slice_uid);
	} else {
	    curr_contour->ct_slice_uid = 0;
	}
        contour_no++;
        curr_structure->num_contours = contour_no;

        //printf ("Gonna dereference curr_contour->x\n");
        curr_contour->x = (float*) malloc (num_pt * sizeof(float));
        curr_contour->y = (float*) malloc (num_pt * sizeof(float));
        curr_contour->z = (float*) malloc (num_pt * sizeof(float));
        if (curr_contour->y == 0 || curr_contour->x == 0) {
            fprintf (stderr, "Error allocating memory");
            exit (-1);
        }
        for (k = 0; k < num_pt; k++) {
	    long floc;
            //printf (" --> (%5d)", k);
	    floc = ftell (fp);
            if (fscanf (fp, "%f\\%f\\%f", &x, &y, &z) != 3) {
		fseek (fp, floc, SEEK_SET);
                if (fscanf (fp, "\\%f\\%f\\%f", &x, &y, &z) != 3) {
		    fseek (fp, floc, SEEK_SET);
		    //printf ("\n", k);
                    break;
                }
            }
            curr_contour->x[k] = x;
            curr_contour->y[k] = y;
            curr_contour->z[k] = z;
	    //printf ("%g %g %g\n", x, y, z);
            x = 0;
            y = 0;
            z = 0;
        }
        slice_idx = 0;
        num_pt = 0;
    }
    fclose (fp);
    printf ("successful!\n");
    return;
    //not_successful:
    //    fclose (fp);
    //    printf ("Error parsing input file.\n");
}

plastimatch1_EXPORT
void
cxt_write (Cxt_structure_list* structures, const char* cxt_fn)
{
    int i;
    FILE *fp;

    fp = fopen (cxt_fn, "wb");
    if (!fp) {
	printf ("Could not open contour file for write: %s\n", cxt_fn);
        exit (-1);
    }

    /* Part 1: Dicom info */
    if (structures->ct_series_uid) {
	fprintf (fp, "CT_SERIES_UID %s\n", structures->ct_series_uid->data);
    } else {
	fprintf (fp, "CT_SERIES_UID\n");
    }
    if (structures->ct_study_uid) {
	fprintf (fp, "CT_STUDY_UID %s\n", structures->ct_study_uid->data);
    } else {
	fprintf (fp, "CT_STUDY_UID\n");
    }
    if (structures->ct_fref_uid) {
	fprintf (fp, "CT_FRAME_OF_REFERENCE_UID %s\n", 
		 structures->ct_fref_uid->data);
    } else {
	fprintf (fp, "CT_FRAME_OF_REFERENCE_UID\n");
    }
    if (structures->patient_name) {
	fprintf (fp, "PATIENT_NAME %s\n", structures->patient_name->data);
    } else {
	fprintf (fp, "PATIENT_NAME\n");
    }
    if (structures->patient_id) {
	fprintf (fp, "PATIENT_ID %s\n", structures->patient_id->data);
    } else {
	fprintf (fp, "PATIENT_ID\n");
    }
    if (structures->patient_sex) {
	fprintf (fp, "PATIENT_SEX %s\n", structures->patient_sex->data);
    } else {
	fprintf (fp, "PATIENT_SEX\n");
    }
    if (structures->patient_sex) {
	fprintf (fp, "STUDY_ID %s\n", structures->study_id->data);
    } else {
	fprintf (fp, "STUDY_ID\n");
    }
    if (structures->have_geometry) {
	fprintf (fp, "OFFSET %g %g %g\n", structures->offset[0],
		 structures->offset[1], structures->offset[2]);
	fprintf (fp, "DIMENSION %d %d %d\n", structures->dim[0], 
		 structures->dim[1], structures->dim[2]);
	fprintf (fp, "SPACING %g %g %g\n", structures->spacing[0], 
		 structures->spacing[1], structures->spacing[2]);
    }

    /* Part 2: Structures info */
    fprintf (fp, "ROI_NAMES\n");
    for (i = 0; i < structures->num_structures; i++) {
	Cxt_structure *curr_structure = &structures->slist[i];
	fprintf (fp, "%d|%s|%s\n", curr_structure->id, "255\\0\\0", curr_structure->name);
    }
    fprintf (fp, "END_OF_ROI_NAMES\n");

    /* Part 3: Contour info */
    for (i = 0; i < structures->num_structures; i++) {
	int j;
	Cxt_structure *curr_structure = &structures->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    int k;
	    Cxt_polyline *curr_polyline = &curr_structure->pslist[j];

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
	    if (curr_polyline->ct_slice_uid) {
		fprintf (fp, "%s|", curr_polyline->ct_slice_uid->data);
	    } else {
		fprintf (fp, "|");
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

plastimatch1_EXPORT
void
cxt_destroy (Cxt_structure_list* structures)
{
    bdestroy (structures->ct_series_uid);
    bdestroy (structures->patient_name);
    bdestroy (structures->patient_id);
    bdestroy (structures->patient_sex);
    bdestroy (structures->study_id);

    /* GCS FIX: This leaks memory */
    memset (structures, 0, sizeof (Cxt_structure_list));
}
