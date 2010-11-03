/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bstring_util.h"
#include "file_util.h"
#include "math_util.h"
#include "plm_image_header.h"
#include "rtss.h"

Rtss::Rtss ()
{
    this->init ();
}

Rtss::~Rtss ()
{
    this->clear ();
}

void
Rtss::init (void)
{
    this->m_demographics = new Demographics;
    this->have_geometry = 0;
    this->num_structures = 0;
    this->slist = 0;
}

void
Rtss::clear (void)
{
    int i;

    this->ct_study_uid = "";
    this->ct_series_uid = "";
    this->ct_fref_uid = "";
    this->study_id = "";
    delete this->m_demographics;

    for (i = 0; i < this->num_structures; i++) {
	delete (this->slist[i]);
    }
    free (this->slist);

    this->init ();
}


/* Add structure (if it doesn't already exist) */
Rtss_structure*
Rtss::add_structure (
    const CBString& structure_name, 
    const CBString& color, 
    int structure_id)
{
    Rtss_structure* new_structure;

    new_structure = this->find_structure_by_id (structure_id);
    if (new_structure) {
	return new_structure;
    }

    this->num_structures++;
    this->slist = (Rtss_structure**) 
	    realloc (this->slist, 
		     this->num_structures * sizeof(Rtss_structure*));
    new_structure 
	= this->slist[this->num_structures - 1] 
	= new Rtss_structure;

    new_structure->name = structure_name;
    new_structure->name.trim();
    new_structure->id = structure_id;
    new_structure->bit = -1;
    new_structure->color = color;
    new_structure->num_contours = 0;
    new_structure->pslist = 0;
    return new_structure;
}

Rtss_structure*
Rtss::find_structure_by_id (int structure_id)
{
    int i;

    for (i = 0; i < this->num_structures; i++) {
	Rtss_structure* curr_structure;
	curr_structure = this->slist[i];
	if (curr_structure->id == structure_id) {
	    return curr_structure;
	}
    }
    return 0;
}

void
Rtss::debug (void)
{
    int i;
    Rtss_structure* curr_structure;

    printf ("dim = %d %d %d\n", 
	this->dim[0], this->dim[1], this->dim[2]);
    printf ("offset = %g %g %g\n", 
	this->offset[0], this->offset[1], this->offset[2]);
    printf ("spacing = %g %g %g\n", 
	this->spacing[0], this->spacing[1], this->spacing[2]);

    for (i = 0; i < this->num_structures; i++) {
        curr_structure = this->slist[i];
	printf ("%d %d %s (%p) (%d contours)", 
	    i, curr_structure->id, 
	    (const char*) curr_structure->name, 
	    curr_structure->pslist, 
	    curr_structure->num_contours
	);
	if (curr_structure->num_contours) {
	    if (curr_structure->pslist[0]->num_vertices) {
		printf (" [%f,%f,%f,...]",
		    curr_structure->pslist[0]->x[0],
		    curr_structure->pslist[0]->y[0],
		    curr_structure->pslist[0]->z[0]);
	    } else {
		printf (" <no vertices>");
	    }
	}
	printf ("\n");
    }
}

void
Rtss::adjust_structure_names (void)
{
    int i, j;
    Rtss_structure* curr_structure;

    for (i = 0; i < this->num_structures; i++) {
        curr_structure = this->slist[i];
	for (j = 0; j < curr_structure->name.length(); j++) {
	    /* GE Adv sim doesn't like names with strange punctuation. */
	    if (! isalnum (curr_structure->name[j])) {
		curr_structure->name[j] = '_';
		printf ("Substituted in name %s\n", 
		    (const char*) curr_structure->name);
	    }
	}
    }
}

void
Rtss::prune_empty (void)
{
    int i;

    for (i = 0; i < this->num_structures; i++) {
	Rtss_structure* curr_structure;
	curr_structure = this->slist[i];
	if (curr_structure->num_contours == 0) {
	    delete curr_structure;
	    /* Remark: the below two lines are correct but redundant if 
	       (i == this->num_structures-1), but this comment to explain 
	       it is not worse than adding if statement. */
	    this->slist[i] = this->slist[this->num_structures-1];
	    i --;
	    this->num_structures --;
	}
    }
}

/* Copy structure name, id, color, but not contents */
Rtss*
Rtss::clone_empty (
    Rtss* cxt_out
)
{
    int i;

    /* Initialize output cxt */
    if (cxt_out) {
	cxt_out->clear ();
    } else {
	cxt_out = new Rtss;
    }

    for (i = 0; i < this->num_structures; i++) {
	Rtss_structure *old_structure = this->slist[i];
	Rtss_structure *new_structure = cxt_out->add_structure (
	    old_structure->name, old_structure->color, old_structure->id);

	/* Copy bit */
	new_structure->bit = old_structure->bit;
    }

    return cxt_out;
}

/* Clear the polylines, but keep structure name, id, color */
void
Rtss::free_all_polylines (void)
{
    int i;
    for (i = 0; i < this->num_structures; i++) {
	int j;
	Rtss_structure *curr_structure = this->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    delete curr_structure->pslist[j];
	}
	free (curr_structure->pslist);

	curr_structure->num_contours = 0;
	curr_structure->pslist = 0;
    }
}

void
Rtss::apply_geometry (void)
{
    int i, j;

    if (!this->have_geometry) return;

    for (i = 0; i < this->num_structures; i++) {
	Rtss_structure *curr_structure = this->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Rtss_polyline *curr_polyline = curr_structure->pslist[j];
	    if (curr_polyline->num_vertices == 0) {
		curr_polyline->slice_no = -1;
		continue;
	    }
	    float z = curr_polyline->z[0];
	    int slice_idx = ROUND_INT((z - this->offset[2]) / this->spacing[2]);
	    if (slice_idx < 0 || slice_idx >= this->dim[2]) {
		curr_polyline->slice_no = -1;
	    } else {
		curr_polyline->slice_no = slice_idx;
	    }
	}
    }
}

void
Rtss::set_geometry_from_plm_image_header (
    Plm_image_header *pih
)
{
    pih->get_gpuit_origin (this->offset);
    pih->get_gpuit_spacing (this->spacing);
    pih->get_gpuit_dim (this->dim);
    this->have_geometry = 1;

    this->apply_geometry ();
}

void
Rtss::set_geometry_from_plm_image (
    Plm_image *pli
)
{
    Plm_image_header pih;
    pih.set_from_plm_image (pli);
    this->set_geometry_from_plm_image_header (&pih);
}
