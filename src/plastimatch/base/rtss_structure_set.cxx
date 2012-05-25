/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <set>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "plmbase.h"
#include "plmsys.h"

#include "plm_math.h"
#include "pstring.h"


static void
assign_random_color (Pstring& color)
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

Rtss_structure_set::Rtss_structure_set ()
{
    this->init ();
}

Rtss_structure_set::~Rtss_structure_set ()
{
    this->clear ();
}

void
Rtss_structure_set::init (void)
{
    this->have_geometry = 0;
    this->num_structures = 0;
    this->slist = 0;
}

void
Rtss_structure_set::clear (void)
{
    for (size_t i = 0; i < this->num_structures; i++) {
	delete (this->slist[i]);
    }
    free (this->slist);

    this->init ();
}

Pstring
Rtss_structure_set::find_unused_structure_name (void)
{
    Pstring test_name;
    for (int n = 1; n < std::numeric_limits<int>::max(); ++n) {
	test_name.format ("%s (%d)", "Unknown structure", n);
	bool dup_found = 0;
	for (size_t i = 0; i < this->num_structures; ++i) {
	    Rtss_structure* curr_structure = this->slist[i];
	    if (test_name == curr_structure->name) {
		dup_found = true;
		break;
	    }
	}
	if (!dup_found) {
	    break;
	}
    }

    return test_name;
}

/* Add structure (if it doesn't already exist) */
Rtss_structure*
Rtss_structure_set::add_structure (
    const Pstring& structure_name, 
    const Pstring& color, 
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
    if (structure_name == "" || structure_name == "Unknown structure") {
	new_structure->name = find_unused_structure_name ();
    }
    new_structure->name.trim();
    new_structure->id = structure_id;
    new_structure->bit = -1;
    if (color.not_empty()) {
	new_structure->color = color;
    } else {
	assign_random_color (new_structure->color);
    }
    new_structure->num_contours = 0;
    new_structure->pslist = 0;
    return new_structure;
}

Rtss_structure*
Rtss_structure_set::find_structure_by_id (int structure_id)
{
    for (size_t i = 0; i < this->num_structures; i++) {
	Rtss_structure* curr_structure;
	curr_structure = this->slist[i];
	if (curr_structure->id == structure_id) {
	    return curr_structure;
	}
    }
    return 0;
}

void
Rtss_structure_set::debug (void)
{
    Rtss_structure* curr_structure;

    if (this->have_geometry) {
	printf ("rps::dim = %u %u %u\n", 
	    (unsigned int) this->m_dim[0], 
	    (unsigned int) this->m_dim[1], 
	    (unsigned int) this->m_dim[2]);
	printf ("rps::offset = %g %g %g\n", 
	    this->m_offset[0], this->m_offset[1], this->m_offset[2]);
	printf ("rps::spacing = %g %g %g\n", 
	    this->m_spacing[0], this->m_spacing[1], this->m_spacing[2]);
    } else {
	printf ("rps has no geometry\n");
    }

    for (size_t i = 0; i < this->num_structures; i++) {
        curr_structure = this->slist[i];
	printf ("%u %d %s [%s] (%p) (%d contours)", 
	    (unsigned int) i, 
	    curr_structure->id, 
	    (const char*) curr_structure->name, 
	    (curr_structure->color.empty() 
                ? "none" : (const char*) curr_structure->color), 
	    curr_structure->pslist, 
	    (int) curr_structure->num_contours
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
Rtss_structure_set::adjust_structure_names (void)
{
    Rtss_structure* curr_structure;

    for (size_t i = 0; i < this->num_structures; i++) {
        curr_structure = this->slist[i];
	bool changed = false;
	Pstring tmp = curr_structure->name;
	for (int j = 0; j < curr_structure->name.length(); j++) {
	    /* GE Adv sim doesn't like names with strange punctuation. */
	    if (! isalnum (curr_structure->name[j])) {
		curr_structure->name[j] = '_';
		changed = true;
	    }
	}
	if (changed && !tmp.has_prefix ("Unknown")) {
	    printf ("Substituted structure name (%s) to (%s)\n", 
		(const char*) tmp, (const char*) curr_structure->name);
	}
    }
}

void
Rtss_structure_set::prune_empty (void)
{
    for (size_t i = 0; i < this->num_structures; i++) {
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
Rtss_structure_set*
Rtss_structure_set::clone_empty (
    Rtss_structure_set* cxt_out,
    Rtss_structure_set* cxt_in
)
{
    /* Initialize output cxt */
    if (cxt_out) {
	cxt_out->clear ();
    } else {
	cxt_out = new Rtss_structure_set;
    }

    for (size_t i = 0; i < cxt_in->num_structures; i++) {
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
Rtss_structure_set::free_all_polylines (void)
{
    for (size_t i = 0; i < this->num_structures; i++) {
	Rtss_structure *curr_structure = this->slist[i];
	for (size_t j = 0; j < curr_structure->num_contours; j++) {
	    delete curr_structure->pslist[j];
	}
	free (curr_structure->pslist);

	curr_structure->num_contours = 0;
	curr_structure->pslist = 0;
    }
}

void
Rtss_structure_set::find_rasterization_geometry (
    float offset[3],
    float spacing[3],
    plm_long dims[3]
)
{
    int first = 1;
    float min_x = 0.f, max_x = 0.f;
    float min_y = 0.f, max_y = 0.f;
    float min_z = 0.f, max_z = 0.f;
    std::set<float> z_values;

    /* Scan points to find image size, spacing */
    for (size_t i = 0; i < this->num_structures; i++) {
	Rtss_structure *curr_structure = this->slist[i];
	for (size_t j = 0; j < curr_structure->num_contours; j++) {
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
Rtss_structure_set::find_rasterization_geometry (Plm_image_header *pih)
{
    /* use some generic default parameters */
    plm_long dim[3];
    float origin[3];
    float spacing[3];

    this->find_rasterization_geometry (origin, spacing, dim);

    pih->set_from_gpuit (dim, origin, spacing, 0);
}

void
Rtss_structure_set::set_rasterization_geometry (void)
{
    this->find_rasterization_geometry (
	this->rast_offset,
	this->rast_spacing,
	this->rast_dim
    );
    printf ("rast_dim = %u %u %u\n", 
	(unsigned int) this->rast_dim[0], 
	(unsigned int) this->rast_dim[1], 
	(unsigned int) this->rast_dim[2]);
    printf ("rast_offset = %g %g %g\n", 
	this->rast_offset[0], this->rast_offset[1], this->rast_offset[2]);
    printf ("rast_spacing = %g %g %g\n", 
	this->rast_spacing[0], this->rast_spacing[1], this->rast_spacing[2]);
}

void
Rtss_structure_set::fix_polyline_slice_numbers (void)
{
    if (!this->have_geometry) return;

    for (size_t i = 0; i < this->num_structures; i++) {
	Rtss_structure *curr_structure = this->slist[i];
	for (size_t j = 0; j < curr_structure->num_contours; j++) {
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
Rtss_structure_set::set_geometry (
    const Plm_image_header *pih
)
{
    pih->get_origin (this->m_offset);
    pih->get_spacing (this->m_spacing);
    pih->get_dim (this->m_dim);
    this->have_geometry = 1;

    this->fix_polyline_slice_numbers ();
}

void
Rtss_structure_set::set_geometry (
    const Plm_image *pli
)
{
    Plm_image_header pih;
    pih.set_from_plm_image (pli);
    this->set_geometry (&pih);
}

void
Rtss_structure_set::keyholize (void)
{
#if defined (PLM_CONFIG_KEYHOLIZE)
    printf ("Keyholizing...\n");

    /* Loop through structures */
    for (int i = 0; i < this->num_structures; i++) {
	Rtss_structure *curr_structure = this->slist[i];

	/* Find groups of contours which lie on the same slice */
	std::vector<bool> used_contours;
	used_contours.assign (curr_structure->num_contours, false);

	for (int j = 0; j < curr_structure->num_contours; j++) {
	    std::vector<int> group_contours;
	    Rtss_polyline *group_polyline = curr_structure->pslist[j];
	    if (group_polyline->num_vertices == 0) {
		group_polyline->slice_no = -1;
		continue;
	    }
	    if (used_contours[j] == true) {
		continue;
	    }
	    float group_z = group_polyline->z[0];
	    group_contours.push_back (j);
	    for (int k = j+1; k < curr_structure->num_contours; k++) {
		Rtss_polyline *curr_polyline = curr_structure->pslist[k];
		if (curr_polyline->num_vertices == 0) {
		    curr_polyline->slice_no = -1;
		    continue;
		}
		float curr_z = curr_polyline->z[0];
		if (curr_z - group_z < SPACING_TOL) {
		    used_contours[k] = true;
		    group_contours.push_back (k);
		}
	    }

	    /* We have now found a group */
	    printf ("Keyholizing group:");
	    for (unsigned int k = 0; k < group_contours.size(); k++) {
		printf (" %d", group_contours[k]);
	    }
	    printf ("\n");

	    /* Find an outermost contour in group */
	    int cidx_xmin = -1;
	    float xmin = FLT_MAX;
	    for (unsigned int k = 0; k < group_contours.size(); k++) {
		int cidx = group_contours[k];
		Rtss_polyline *curr_polyline = curr_structure->pslist[cidx];

		float curr_xmin = FLT_MAX;
		for (int l = 0; l < curr_polyline->num_vertices; l++) {
		    if (curr_polyline->x[l] < curr_xmin) {
			curr_xmin = curr_polyline->x[l];
		    }
		}

		if (curr_xmin < xmin) {
		    cidx_xmin = cidx;
		    xmin = curr_xmin;
		}
	    }
	    
	    /* Loop through other contours, find contours contained 
	       in this contour */
	    for (unsigned int k = 0; k < group_contours.size(); k++) {
		int cidx = group_contours[k];
		Rtss_polyline *curr_polyline = curr_structure->pslist[cidx];
		if (cidx == cidx_xmin) {
		    continue;
		}

		float x = curr_polyline->x[0];
		float y = curr_polyline->y[0];
		
	    }
	}
    }
#endif
}