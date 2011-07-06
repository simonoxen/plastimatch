/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bstring_util.h"
#include "file_util.h"
#include "img_metadata.h"
#include "math_util.h"
#include "plm_image_header.h"
#include "rtss_polyline_set.h"
#include "rtss_structure.h"


static void
assign_random_color (CBString& color)
{
    static int idx = 0;
    static const char* colors[] = {
	"255 0 0",
	"255 255 0",
	"255 0 255",
	"0 255 255",
	"0 255 0",
	"0 0 255",
	"255 128 128",
	"255 255 128",
	"255 128 255",
	"128 255 255",
	"128 255 128",
	"128 128 255",
	"200 128 128",
	"200 200 128",
	"200 128 200",
	"128 200 200",
	"128 200 128",
	"128 128 200",
	"200 255 255",
	"200 200 255",
	"200 255 200",
	"255 200 200",
	"255 200 255",
	"255 255 200",
    };
    color = colors[idx];
    if (++idx > 23) {
	idx = 0;
    }
}

#define SPACING_TOL 0.2    /* How close you need to be to be on the slice */

Rtss_polyline_set::Rtss_polyline_set ()
{
    this->init ();
}

Rtss_polyline_set::~Rtss_polyline_set ()
{
    this->clear ();
}

void
Rtss_polyline_set::init (void)
{
    this->have_geometry = 0;
    this->num_structures = 0;
    this->slist = 0;
}

void
Rtss_polyline_set::clear (void)
{
    int i;

    for (i = 0; i < this->num_structures; i++) {
	delete (this->slist[i]);
    }
    free (this->slist);

    this->init ();
}


/* Add structure (if it doesn't already exist) */
Rtss_structure*
Rtss_polyline_set::add_structure (
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
    if (bstring_not_empty (color)) {
	new_structure->color = color;
    } else {
	assign_random_color (new_structure->color);
    }
    new_structure->num_contours = 0;
    new_structure->pslist = 0;
    return new_structure;
}

Rtss_structure*
Rtss_polyline_set::find_structure_by_id (int structure_id)
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
Rtss_polyline_set::debug (void)
{
    int i;
    Rtss_structure* curr_structure;

    printf ("dim = %d %d %d\n", 
	this->m_dim[0], this->m_dim[1], this->m_dim[2]);
    printf ("offset = %g %g %g\n", 
	this->m_offset[0], this->m_offset[1], this->m_offset[2]);
    printf ("spacing = %g %g %g\n", 
	this->m_spacing[0], this->m_spacing[1], this->m_spacing[2]);

    for (i = 0; i < this->num_structures; i++) {
        curr_structure = this->slist[i];
	printf ("%d %d %s [%s] (%p) (%d contours)", 
	    i, 
	    curr_structure->id, 
	    (const char*) curr_structure->name, 
	    bstring_empty (curr_structure->color) 
	      ? "none" : (const char*) curr_structure->color, 
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
Rtss_polyline_set::adjust_structure_names (void)
{
    int i, j;
    Rtss_structure* curr_structure;

    for (i = 0; i < this->num_structures; i++) {
        curr_structure = this->slist[i];
	bool changed = false;
	CBString tmp = curr_structure->name;
	for (j = 0; j < curr_structure->name.length(); j++) {
	    /* GE Adv sim doesn't like names with strange punctuation. */
	    if (! isalnum (curr_structure->name[j])) {
		curr_structure->name[j] = '_';
		changed = true;
	    }
	}
	if (changed) {
	    printf ("Substituted structure name (%s) to (%s)\n", 
		(const char*) tmp, (const char*) curr_structure->name);
	}
    }
}

void
Rtss_polyline_set::prune_empty (void)
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
Rtss_polyline_set*
Rtss_polyline_set::clone_empty (
    Rtss_polyline_set* cxt_out,
    Rtss_polyline_set* cxt_in
)
{
    int i;

    /* Initialize output cxt */
    if (cxt_out) {
	cxt_out->clear ();
    } else {
	cxt_out = new Rtss_polyline_set;
    }

    for (i = 0; i < cxt_in->num_structures; i++) {
	Rtss_structure *old_structure = cxt_in->slist[i];
	Rtss_structure *new_structure = cxt_out->add_structure (
	    old_structure->name, old_structure->color, old_structure->id);

	/* Copy bit */
	new_structure->bit = old_structure->bit;
    }
    return cxt_out;
}

/* Clear the polylines, but keep structure name, id, color */
void
Rtss_polyline_set::free_all_polylines (void)
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
Rtss_polyline_set::find_rasterization_geometry (
    float offset[3],
    float spacing[3],
    int dims[3]
)
{
    int first = 1;
    float min_x = 0.f, max_x = 0.f;
    float min_y = 0.f, max_y = 0.f;
    float min_z = 0.f, max_z = 0.f;
    std::set<float> z_values;

    /* Scan points to find image size, spacing */
    for (int i = 0; i < this->num_structures; i++) {
	Rtss_structure *curr_structure = this->slist[i];
	for (int j = 0; j < curr_structure->num_contours; j++) {
	    Rtss_polyline *curr_polyline = curr_structure->pslist[j];
	    for (int k = 0; k < curr_polyline->num_vertices; k++) {
		z_values.insert (curr_polyline->z[k]);
		if (first) {
		    min_x = max_x = curr_polyline->x[k];
		    min_y = max_y = curr_polyline->y[k];
		    min_z = max_z = curr_polyline->z[k];
		    first = 0;
		    continue;
		}
		if (curr_polyline->x[k] < min_x) {
		    min_x = curr_polyline->x[k];
		} else if (curr_polyline->x[k] > max_x) {
		    max_x = curr_polyline->x[k];
		}
		if (curr_polyline->y[k] < min_y) {
		    min_y = curr_polyline->y[k];
		} else if (curr_polyline->y[k] > max_y) {
		    max_y = curr_polyline->y[k];
		}
		if (curr_polyline->z[k] < min_z) {
		    min_z = curr_polyline->z[k];
		} else if (curr_polyline->z[k] > max_z) {
		    max_z = curr_polyline->z[k];
		}
	    }
	}
    }

    /* Use heuristics to set (x,y) values */
    float range_x = max_x - min_x;
    float range_y = max_y - min_y;
    float range = range_x;
    if (range_y > range) {
	range = range_y;
    }
    range = range * 1.05;
    spacing[0] = spacing[1] = range / 512;
    offset[0] = 0.5 * (max_x + min_x - range);
    offset[1] = 0.5 * (max_y + min_y - range);
    dims[0] = dims[1] = 512;

#if defined (commentout)
    printf ("----Z VALUES-----\n");
    for (std::set<float>::iterator it = z_values.begin(); 
	 it != z_values.end(); 
	 it++)
    {
	printf ("%f\n", *it);
    }
    printf ("---------\n");
#endif

    /* z value should be based on native slice spacing */
    int have_spacing = 0;
    float z_spacing = 0.f;
    float last_z = min_z;
    for (std::set<float>::iterator it = z_values.begin(); 
	 it != z_values.end(); 
	 it++)
    {
	float this_z = *it;
	float diff = this_z - last_z;
	if (fabs (diff) < SPACING_TOL) {
	    continue;
	}
	if (!have_spacing) {
	    z_spacing = this_z - min_z;
	    have_spacing = 1;
	} else {
	    if (fabs (diff - z_spacing) > SPACING_TOL) {
		printf ("Warning, slice spacing of RTSS may be unequal\n");
		printf ("%g - %g = %g vs. %g\n", 
		    this_z, last_z, diff, z_spacing);
	    }
	}
	last_z = this_z;
    }
    
    offset[2] = min_z;
    if (have_spacing) {
	dims[2] = ROUND_INT ((max_z - min_z) / z_spacing);
	spacing[2] = z_spacing;
    } else {
	dims[2] = 1;
	spacing[2] = 1;
    }
}

void
Rtss_polyline_set::find_rasterization_geometry (Plm_image_header *pih)
{
    /* use some generic default parameters */
    int dim[3];
    float origin[3];
    float spacing[3];

    this->find_rasterization_geometry (origin, spacing, dim);

    pih->set_from_gpuit (dim, origin, spacing, 0);
}

void
Rtss_polyline_set::set_rasterization_geometry (void)
{
    this->find_rasterization_geometry (
	this->rast_offset,
	this->rast_spacing,
	this->rast_dim
    );
    printf ("rast_dim = %d %d %d\n", 
	this->rast_dim[0], this->rast_dim[1], this->rast_dim[2]);
    printf ("rast_offset = %g %g %g\n", 
	this->rast_offset[0], this->rast_offset[1], this->rast_offset[2]);
    printf ("rast_spacing = %g %g %g\n", 
	this->rast_spacing[0], this->rast_spacing[1], this->rast_spacing[2]);
}

void
Rtss_polyline_set::fix_polyline_slice_numbers (void)
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
	    int slice_idx = ROUND_INT (
		(z - this->m_offset[2]) / this->m_spacing[2]);
	    if (slice_idx < 0 || slice_idx >= this->m_dim[2]) {
		curr_polyline->slice_no = -1;
	    } else {
		curr_polyline->slice_no = slice_idx;
	    }
	}
    }
}

void
Rtss_polyline_set::set_geometry_from_plm_image_header (
    Plm_image_header *pih
)
{
    pih->get_origin (this->m_offset);
    pih->get_spacing (this->m_spacing);
    pih->get_dim (this->m_dim);
    this->have_geometry = 1;

    this->fix_polyline_slice_numbers ();
}

void
Rtss_polyline_set::set_geometry_from_plm_image (
    Plm_image *pli
)
{
    Plm_image_header pih;
    pih.set_from_plm_image (pli);
    this->set_geometry_from_plm_image_header (&pih);
}

void
Rtss_polyline_set::keyholize (void)
{

}
