/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bstrlib.h"
#include "cxt.h"
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

/* Add structure (if it doesn't already exist) */
Cxt_structure*
cxt_add_structure (Cxt_structure_list* cxt, const char *structure_name,
    bstring color, int structure_id)
{
    Cxt_structure* new_structure;

    new_structure = cxt_find_structure_by_id (cxt, structure_id);
    if (new_structure) {
	return new_structure;
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
    new_structure->bit = -1;
    new_structure->color = color;
    new_structure->num_contours = 0;
    new_structure->pslist = 0;
    return new_structure;
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
	printf ("%d %d %s (%p) (%d contours)", 
	    i, curr_structure->id, 
	    curr_structure->name, 
	    curr_structure->pslist,
	    curr_structure->num_contours
	);
	if (curr_structure->num_contours) {
	    if (curr_structure->pslist[0].num_vertices) {
		printf (" [%f,%f,%f,...]",
		    curr_structure->pslist[0].x[0],
		    curr_structure->pslist[0].y[0],
		    curr_structure->pslist[0].z[0]);
	    } else {
		printf (" <no vertices>");
	    }
	}
	printf ("\n");
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

/* Copy structure name, id, color, but not contents */
Cxt_structure_list*
cxt_clone_empty (
    Cxt_structure_list* cxt_out, 
    Cxt_structure_list* cxt_in
)
{
    int i;

    /* Initialize output cxt */
    if (cxt_out) {
	cxt_free (cxt_out);
    } else {
	cxt_out = cxt_create ();
    }

    for (i = 0; i < cxt_in->num_structures; i++) {
	Cxt_structure *old_structure = &cxt_in->slist[i];
	Cxt_structure *new_structure = cxt_add_structure (
	    cxt_out, old_structure->name,
	    old_structure->color, old_structure->id);

	/* Copy bit */
	new_structure->bit = old_structure->bit;
    }

    return cxt_out;
}

/* Clear the polylines, but keep structure name, id, color */
void
cxt_free_all_polylines (Cxt_structure_list* cxt)
{
    int i;
    for (i = 0; i < cxt->num_structures; i++) {
	int j;
	Cxt_structure *curr_structure = &cxt->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    cxt_polyline_free (&curr_structure->pslist[j]);
	}
	free (curr_structure->pslist);

	curr_structure->num_contours = 0;
	curr_structure->pslist = 0;
    }
}

void
cxt_free (Cxt_structure_list* cxt)
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
cxt_destroy (Cxt_structure_list* cxt)
{
    cxt_free (cxt);
    free (cxt);
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
    Plm_image_header *pih
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
    Plm_image *pli
)
{
    Plm_image_header pih;
    pih.set_from_plm_image (pli);
    cxt_set_geometry_from_plm_image_header (cxt, &pih);
}
