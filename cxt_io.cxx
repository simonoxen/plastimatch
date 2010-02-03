/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bstrlib.h"
#include "cxt_io.h"
#include "file_util.h"
#include "math_util.h"
#include "plm_image_header.h"

Cxt_structure_list*
cxt_create (void)
{
    Cxt_structure_list *cxt;
    cxt = (Cxt_structure_list*) malloc (sizeof (Cxt_structure_list));
    cxt_init (cxt);
    return cxt;
}

void
cxt_init (Cxt_structure_list* cxt)
{
    memset (cxt, 0, sizeof (Cxt_structure_list));
}

void
cxt_add_structure (Cxt_structure_list* cxt, const char *structure_name,
		   bstring color, int structure_id)
{
    Cxt_structure* new_structure;

    if (cxt_find_structure_by_id (cxt, structure_id)) {
	return;
    }

    cxt->num_structures++;
    cxt->slist = (Cxt_structure*) 
	    realloc (cxt->slist, 
		     cxt->num_structures * sizeof(Cxt_structure));
    new_structure = &cxt->slist[cxt->num_structures - 1];

    memset (new_structure, 0, sizeof(Cxt_structure));
    strncpy (new_structure->name, structure_name, CXT_BUFLEN);
    new_structure->name[CXT_BUFLEN-1] = 0;
    new_structure->id = structure_id;
    new_structure->color = color;
    new_structure->num_contours = 0;
    new_structure->pslist = 0;
}

Cxt_polyline*
cxt_add_polyline (Cxt_structure* structure)
{
    Cxt_polyline* new_polyline;

    structure->num_contours++;
    structure->pslist = (Cxt_polyline*) 
	realloc (structure->pslist, structure->num_contours 
		 * sizeof(Cxt_polyline));

    new_polyline = &structure->pslist[structure->num_contours - 1];
    memset (new_polyline, 0, sizeof(Cxt_polyline));
    return new_polyline;
}

Cxt_structure*
cxt_find_structure_by_id (Cxt_structure_list* cxt, int structure_id)
{
    int i;

    for (i = 0; i < cxt->num_structures; i++) {
	Cxt_structure* curr_structure;
	curr_structure = &cxt->slist[i];
	if (curr_structure->id == structure_id) {
	    return curr_structure;
	}
    }
    return 0;
}

void
cxt_debug (Cxt_structure_list* cxt)
{
    int i;
    Cxt_structure* curr_structure;

    printf ("dim = %d %d %d\n", 
	cxt->dim[0], cxt->dim[1], cxt->dim[2]);
    printf ("offset = %g %g %g\n", 
	cxt->offset[0], cxt->offset[1], cxt->offset[2]);
    printf ("spacing = %g %g %g\n", 
	cxt->spacing[0], cxt->spacing[1], cxt->spacing[2]);

    for (i = 0; i < cxt->num_structures; i++) {
        curr_structure = &cxt->slist[i];
	printf ("%d %d %s (%p) (%d contours)\n", 
	    i, curr_structure->id, 
	    curr_structure->name, 
	    curr_structure->pslist,
	    curr_structure->num_contours
	);
    }
}

void
cxt_adjust_structure_names (Cxt_structure_list* cxt)
{
    int i, j;
    Cxt_structure* curr_structure;

    for (i = 0; i < cxt->num_structures; i++) {
        curr_structure = &cxt->slist[i];
	for (j = 0; j < CXT_BUFLEN; j++) {
	    if (!curr_structure->name[j]) {
		break;
	    }

	    /* GE Adv sim doesn't like names with strange punctuation. */
	    if (! isalnum (curr_structure->name[j])) {
		curr_structure->name[j] = '_';
		printf ("Substituted in name %s\n", curr_structure->name);
	    }
	}
    }
}

void
cxt_xorlist_read (Cxt_structure_list* cxt, const char* xorlist_fn)
{
    FILE* fp;

    fp = fopen (xorlist_fn, "r");

    if (!fp) {
	fprintf (stderr, 
		 "Could not open xorlist file for read: %s\n", xorlist_fn);
        exit (-1);
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
	    break;
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_id, color, name);
        if (rc != 3) {
            fprintf (stderr, 
		     "Error. xorlist file not formatted correctly: %s\n",
		     xorlist_fn);
            exit (-1);
        }

	cxt_add_structure (cxt, name, bfromcstr (color), struct_id);
    }

    fclose (fp);
}

void
cxt_read (Cxt_structure_list* cxt, const char* cxt_fn)
{
    FILE* fp;
    Cxt_polyline* curr_contour;

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
        else if (biseqcstr (tag, "PATIENT_NAME")) {
	    cxt->patient_name = bstrcpy (val);
	}
        else if (biseqcstr (tag, "PATIENT_ID")) {
	    cxt->patient_id = bstrcpy (val);
	}
        else if (biseqcstr (tag, "PATIENT_SEX")) {
	    cxt->patient_sex = bstrcpy (val);
	}
        else if (biseqcstr (tag, "STUDY_ID")) {
	    cxt->study_id = bstrcpy (val);
	}
        else if (biseqcstr (tag, "CT_STUDY_UID")) {
	    cxt->ct_study_uid = bstrcpy (val);
	}
        else if (biseqcstr (tag, "CT_SERIES_UID")) {
	    cxt->ct_series_uid = bstrcpy (val);
	}
        else if (biseqcstr (tag, "CT_FRAME_OF_REFERENCE_UID")) {
	    cxt->ct_fref_uid = bstrcpy (val);
	}
        else if (biseqcstr (tag, "OFFSET")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", &val_x, &val_y, &val_z)) {
		have_offset = 1;
		cxt->offset[0] = val_x;
		cxt->offset[1] = val_y;
		cxt->offset[2] = val_z;
	    }
	}
        else if (biseqcstr (tag, "DIMENSION")) {
	    int int_x, int_y, int_z;

	    if (3 == sscanf ((const char*) val->data, "%d %d %d", 
			     &int_x, &int_y, &int_z)) {
		have_dim = 1;
		cxt->dim[0] = int_x;
		cxt->dim[1] = int_y;
		cxt->dim[2] = int_z;
	    }
	}
        else if (biseqcstr (tag, "SPACING")) {
	    if (3 == sscanf ((const char*) val->data, "%f %f %f", &val_x, &val_y, &val_z)) {
		have_spacing = 1;
		cxt->spacing[0] = val_x;
		cxt->spacing[1] = val_y;
		cxt->spacing[2] = val_z;
	    }
	}
	bdestroy (tag);
	bdestroy (val);
    }
    if (have_offset && have_dim && have_spacing) {
	cxt->have_geometry = 1;
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
            fprintf (stderr, 
		     "Error. cxt file is not formatted correctly: %s\n",
		     cxt_fn);
            exit (-1);
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_id, color, name);
        if (rc != 3) {
            break;
        }

#if defined (commentout)
        structures->num_structures++;
        structures->slist = (Cxt_structure*) 
		realloc (structures->slist,
			 structures->num_structures * sizeof(Cxt_structure));

        curr_structure = &structures->slist[structures->num_structures - 1];
        strcpy (curr_structure->name, name);

        curr_structure->id = struct_id;
        curr_structure->color = bfromcstr (color);
        curr_structure->num_contours = 0;
        curr_structure->pslist = 0;
        printf ("Cxt_structure: %s\n", curr_structure->name);
#endif

	cxt_add_structure (cxt, name, bfromcstr (color), struct_id);
    }

    /* Part 3: Contour info */
    while (1) {
	int k;
	int num_pt;
	int struct_id = 0;
	int slice_idx;
	char slice_uid[1024];
	Cxt_structure* curr_structure;

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

        curr_structure = cxt_find_structure_by_id (cxt, struct_id);

	/* If there is no header line for this structure, we will 
	   skip all contours for the structure. */
	if (!curr_structure) {
	    /* Skip to end of line */
	    while (fgetc (fp) != '\n') ;
	    continue;
	}

	++ (curr_structure->num_contours);
        curr_structure->pslist = (Cxt_polyline*) 
		realloc (curr_structure->pslist, 
			 curr_structure->num_contours * sizeof(Cxt_polyline));
	
        curr_contour = &curr_structure->pslist[curr_structure->num_contours-1];
        curr_contour->num_vertices = num_pt;
        curr_contour->slice_no = slice_idx;
	if (slice_uid[0]) {
	    curr_contour->ct_slice_uid = bfromcstr (slice_uid);
	} else {
	    curr_contour->ct_slice_uid = 0;
	}

        curr_contour->x = (float*) malloc (num_pt * sizeof(float));
        curr_contour->y = (float*) malloc (num_pt * sizeof(float));
        curr_contour->z = (float*) malloc (num_pt * sizeof(float));
        if (curr_contour->y == 0 || curr_contour->x == 0) {
            fprintf (stderr, "Error allocating memory");
            exit (-1);
        }
        for (k = 0; k < num_pt; k++) {
	    long floc;
	    floc = ftell (fp);
            if (fscanf (fp, "%f\\%f\\%f\\", &x, &y, &z) != 3) {
#if defined (commentout)
		fseek (fp, floc, SEEK_SET);
                if (fscanf (fp, "\\%f\\%f\\%f", &x, &y, &z) != 3) {
		    char buf1[2048];
		    fseek (fp, floc, SEEK_SET);
		    fread (buf1, 1, 2047, fp);
		    buf1[2047] = 0;
		    printf ("\nBUF\n%s\n", buf1);
		    exit (1);
                    break;
                }
		break;
#endif
		goto not_successful;
            }
            curr_contour->x[k] = x;
            curr_contour->y[k] = y;
            curr_contour->z[k] = z;
	    //printf ("[%g %g %g] ", x, y, z);
            x = 0;
            y = 0;
            z = 0;
        }
	//printf ("|%d\n", k); fflush (stdout);
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
cxt_prune_empty (Cxt_structure_list* cxt)
{
    int i;

    for (i = 0; i < cxt->num_structures; i++) {
	Cxt_structure* curr_structure;
	curr_structure = &cxt->slist[i];
	if (curr_structure->num_contours == 0) {
	    memcpy (curr_structure, 
		&cxt->slist[cxt->num_structures-1],
		sizeof (Cxt_structure));
	    cxt->num_structures --;
	    i --;
	}
    }
}

void
cxt_write (Cxt_structure_list* cxt, const char* cxt_fn,
	   bool prune_empty)
{
    int i;
    FILE *fp;

    /* Prepare output directory */
    make_directory_recursive (cxt_fn);

    fp = fopen (cxt_fn, "wb");
    if (!fp) {
	fprintf (stderr, 
		 "Could not open contour file for write: %s\n", cxt_fn);
        exit (-1);
    }

    /* Part 1: Dicom info */
    if (cxt->ct_series_uid) {
	fprintf (fp, "CT_SERIES_UID %s\n", cxt->ct_series_uid->data);
    } else {
	fprintf (fp, "CT_SERIES_UID\n");
    }
    if (cxt->ct_study_uid) {
	fprintf (fp, "CT_STUDY_UID %s\n", cxt->ct_study_uid->data);
    } else {
	fprintf (fp, "CT_STUDY_UID\n");
    }
    if (cxt->ct_fref_uid) {
	fprintf (fp, "CT_FRAME_OF_REFERENCE_UID %s\n", 
		 cxt->ct_fref_uid->data);
    } else {
	fprintf (fp, "CT_FRAME_OF_REFERENCE_UID\n");
    }
    if (cxt->patient_name) {
	fprintf (fp, "PATIENT_NAME %s\n", cxt->patient_name->data);
    } else {
	fprintf (fp, "PATIENT_NAME\n");
    }
    if (cxt->patient_id) {
	fprintf (fp, "PATIENT_ID %s\n", cxt->patient_id->data);
    } else {
	fprintf (fp, "PATIENT_ID\n");
    }
    if (cxt->patient_sex) {
	fprintf (fp, "PATIENT_SEX %s\n", cxt->patient_sex->data);
    } else {
	fprintf (fp, "PATIENT_SEX\n");
    }
    if (cxt->patient_sex) {
	fprintf (fp, "STUDY_ID %s\n", cxt->study_id->data);
    } else {
	fprintf (fp, "STUDY_ID\n");
    }
    if (cxt->have_geometry) {
	fprintf (fp, "OFFSET %g %g %g\n", cxt->offset[0],
		 cxt->offset[1], cxt->offset[2]);
	fprintf (fp, "DIMENSION %d %d %d\n", cxt->dim[0], 
		 cxt->dim[1], cxt->dim[2]);
	fprintf (fp, "SPACING %g %g %g\n", cxt->spacing[0], 
		 cxt->spacing[1], cxt->spacing[2]);
    }

    /* Part 2: Structures info */
    fprintf (fp, "ROI_NAMES\n");
    for (i = 0; i < cxt->num_structures; i++) {
	Cxt_structure *curr_structure = &cxt->slist[i];
	if (prune_empty && curr_structure->num_contours <= 0) {
	    continue;
	}
	fprintf (fp, "%d|%s|%s\n", curr_structure->id, 
		 curr_structure->color 
		 ? (const char*) curr_structure->color->data : "255\\0\\0", 
		 curr_structure->name);
    }
    fprintf (fp, "END_OF_ROI_NAMES\n");

    /* Part 3: Contour info */
    for (i = 0; i < cxt->num_structures; i++) {
	int j;
	Cxt_structure *curr_structure = &cxt->slist[i];
	if (prune_empty && curr_structure->num_contours <= 0) {
	    continue;
	}
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

static void
cxt_polyline_free (Cxt_polyline* polyline)
{
    bdestroy (polyline->ct_slice_uid);
    free (polyline->x);
    free (polyline->y);
    free (polyline->z);

    polyline->slice_no = -1;
    polyline->ct_slice_uid = 0;
    polyline->num_vertices = 0;
    polyline->x = 0;
    polyline->y = 0;
    polyline->z = 0;
}

static void
cxt_structure_free (Cxt_structure* structure)
{
    int i;
    bdestroy (structure->color);
    for (i = 0; i < structure->num_contours; i++) {
	cxt_polyline_free (&structure->pslist[i]);
    }
    free (structure->pslist);

    structure->name[0] = 0;
    structure->color = 0;
    structure->id = -1;
    structure->num_contours = 0;
    structure->pslist = 0;
}

void
cxt_destroy (Cxt_structure_list* cxt)
{
    int i;

    bdestroy (cxt->ct_series_uid);
    bdestroy (cxt->patient_name);
    bdestroy (cxt->patient_id);
    bdestroy (cxt->patient_sex);
    bdestroy (cxt->study_id);

    for (i = 0; i < cxt->num_structures; i++) {
	cxt_structure_free (&cxt->slist[i]);
    }
    free (cxt->slist);

    cxt_init (cxt);
}

void
cxt_apply_geometry (Cxt_structure_list* cxt)
{
    int i, j;

    if (!cxt->have_geometry) return;

    for (i = 0; i < cxt->num_structures; i++) {
	Cxt_structure *curr_structure = &cxt->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Cxt_polyline *curr_polyline = &curr_structure->pslist[j];
	    if (curr_polyline->num_vertices == 0) {
		curr_polyline->slice_no = -1;
		continue;
	    }
	    float z = curr_polyline->z[0];
	    int slice_idx = ROUND_INT((z - cxt->offset[2]) / cxt->spacing[2]);
	    if (slice_idx < 0 || slice_idx >= cxt->dim[2]) {
		curr_polyline->slice_no = -1;
	    } else {
		curr_polyline->slice_no = slice_idx;
	    }
	}
    }
}

void
cxt_set_geometry_from_plm_image_header (
    Cxt_structure_list* cxt,
    PlmImageHeader *pih
)
{
    pih->get_gpuit_origin (cxt->offset);
    pih->get_gpuit_spacing (cxt->spacing);
    pih->get_gpuit_dim (cxt->dim);
    cxt->have_geometry = 1;

    cxt_apply_geometry (cxt);
}

void
cxt_set_geometry_from_plm_image (
    Cxt_structure_list* cxt, 
    PlmImage *pli
)
{
    PlmImageHeader pih;
    pih.set_from_plm_image (pli);
    cxt_set_geometry_from_plm_image_header (cxt, &pih);
}
