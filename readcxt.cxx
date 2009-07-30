/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "readcxt.h"

plastimatch1_EXPORT
void
cxt_initialize (Cxt_structure_list* structures)
{
    memset (structures, 0, sizeof (Cxt_structure_list));
}

plastimatch1_EXPORT
void
cxt_add_structure (Cxt_structure_list* structures, const char *structure_name, int structure_id)
{
    Cxt_structure* new_structure;

    structures->num_structures++;
    structures->slist = (Cxt_structure*) realloc (structures->slist,
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
    int num_pt = 0;
    int old_struct_no = -1;
    int contour_no = 0;
    int slice_idx = -1;
    char tag[CXT_BUFLEN];

    float x = 0;
    float y = 0;
    float z = 0;

    memset (curr_structure, 0, sizeof(Cxt_structure));
    curr_structure->num_contours = 0;

    fp = fopen (cxt_fn, "r");

    if (!fp) {
	printf ("Could not open contour file for read: %s\n", cxt_fn);
        exit (-1);
    }

    printf ("Loading...\n");
    while (1) {
        char buf[CXT_BUFLEN];
        char *p;

        p = fgets (buf, CXT_BUFLEN, fp);
        if (!p) {
            fprintf (stderr, "ERROR: Your file is not formatted correctly!\n");
            exit (-1);
        }
        if (!strncmp (buf, "ROI_NAMES", strlen ("ROI_NAMES"))) {
            break;
        }
        if (4 == sscanf (buf, "%s %f %f %f", tag, &val_x, &val_y, &val_z)) {
            if (strcmp ("OFFSET", tag) == 0) {
                structures->offset[0] = val_x;
                structures->offset[1] = val_y;
                structures->offset[2] = val_z;
                //printf("%s\n",tag);
            } else if (strcmp ("DIMENSION", tag) == 0) {
                structures->dim[0] = val_x;
                structures->dim[1] = val_y;
                structures->dim[2] = val_z;
                //printf("%s\n",tag);
            } else if (strcmp ("SPACING", tag) == 0) {
                structures->spacing[0] = val_x;
                structures->spacing[1] = val_y;
                structures->spacing[2] = val_z;
                //printf("%s\n",tag);
            }
        }
    }
    while (1) {
        char color[CXT_BUFLEN];
        char name[CXT_BUFLEN];
        char buf[CXT_BUFLEN];
        char *p;
        int rc;

        p = fgets (buf, CXT_BUFLEN, fp);
        if (!p) {
            fprintf (stderr, "ERROR: Your file is not formatted correctly!\n");
            exit (-1);
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_no, color, name);
        if (rc != 3) {
            break;
        }

        structures->num_structures++;
        structures->slist = (Cxt_structure*) realloc (structures->slist,
                                                  structures->num_structures * sizeof(Cxt_structure));
        curr_structure = &structures->slist[structures->num_structures - 1];
        strcpy (curr_structure->name, name);
        curr_structure->num_contours = 0;
        curr_structure->pslist = 0;
        printf ("Cxt_structure: %s\n", curr_structure->name);
    }

    while (1) {
	int k;

        if (1 != fscanf (fp, " %d", &struct_no)) {
	    /* Normal exit from loop */
	    break;
        }
        fgetc (fp);

        /* Skip contour thickness */
        while (fgetc (fp) != '|') ;

        if (1 != fscanf (fp, "%d", &num_pt)) {
	    goto not_successful;
        }
        fgetc (fp);

        if (1 != fscanf (fp, "%d", &slice_idx)) {
	    goto not_successful;
        }
        fgetc (fp);

        /* Skip uid */
        while (fgetc (fp) != '|') ;

        //printf ("%d %d %d\n", struct_no, num_pt, slice_idx);

        if (struct_no != old_struct_no) {
            old_struct_no = struct_no;
            contour_no = 0;
        }
        curr_structure = &structures->slist[struct_no - 1];
        //printf ("Gonna realloc %p, %d\n", curr_structure->pslist, contour_no);
        curr_structure->pslist = (Cxt_polyline*) realloc (curr_structure->pslist,
                                                      (contour_no + 1) * sizeof(Cxt_polyline));
        //printf ("Gonna dereference pslist\n");
        curr_contour = &curr_structure->pslist[contour_no];
        curr_contour->num_vertices = num_pt;
        curr_contour->slice_no = slice_idx;
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
not_successful:
    fclose (fp);
    printf ("Error parsing input file.\n");
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
    fprintf (fp, "SERIES_CT_UID\n");

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
	    fprintf (fp, "%d||%d|||", curr_structure->id, curr_polyline->num_vertices);
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
