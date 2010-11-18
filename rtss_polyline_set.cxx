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
#include "rtss_polyline_set.h"

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
    this->m_demographics = new Demographics;
    this->have_geometry = 0;
    this->num_structures = 0;
    this->slist = 0;
}

void
Rtss_polyline_set::clear (void)
{
    int i;

    this->ct_study_uid = "";
    this->ct_series_uid = "";
    this->ct_fref_uid = "";
    this->study_id = "";
    this->ct_slice_uids.clear();
    delete this->m_demographics;

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
    new_structure->color = color;
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
Rtss_polyline_set::adjust_structure_names (void)
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
Rtss_polyline_set::apply_geometry (void)
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

#if defined (commentout)  /* To be written... - GCS */
void
cxt_apply_dicom_dir (Rtss_polyline_set *cxt, const char *dicom_dir)
{
    int i, j;
    Gdcm_series gs;
    std::string tmp;

    if (!dicom_dir) {
	return;
    }

    gs.load (dicom_dir);
    gs.digest_files ();
    if (!gs.m_have_ct) {
	return;
    }
    gdcm::File* file = gs.get_ct_slice ();

    /* Add geometry */
    int d;
    cxt->have_geometry = 1;
    for (d = 0; d < 3; d++) {
	cxt->offset[d] = gs.m_origin[d];
	cxt->dim[d] = gs.m_dim[d];
	cxt->spacing[d] = gs.m_spacing[d];
    }

    /* PatientName */
    tmp = file->GetEntryValue (0x0010, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->m_demographics->m_patient_name = tmp.c_str();
    }

    /* PatientID */
    tmp = file->GetEntryValue (0x0010, 0x0020);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->m_demographics->m_patient_id = tmp.c_str();
    }

    /* PatientSex */
    tmp = file->GetEntryValue (0x0010, 0x0040);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->m_demographics->m_patient_sex = tmp.c_str();
    }

    /* StudyID */
    tmp = file->GetEntryValue (0x0020, 0x0010);
    if (tmp != gdcm::GDCM_UNFOUND) {
	cxt->study_id = tmp.c_str();
    }

    /* StudyInstanceUID */
    tmp = file->GetEntryValue (0x0020, 0x000d);
    cxt->ct_study_uid = tmp.c_str();

    /* SeriesInstanceUID */
    tmp = file->GetEntryValue (0x0020, 0x000e);
    cxt->ct_series_uid = tmp.c_str();
	
    /* FrameOfReferenceUID */
    tmp = file->GetEntryValue (0x0020, 0x0052);
    cxt->ct_fref_uid = tmp.c_str();

    /* Slice uids */
    gs.get_slice_uids (&cxt->ct_slice_uids);

    /* Slice numbers and slice uids */
    for (i = 0; i < cxt->num_structures; i++) {
	Rtss_structure *curr_structure = cxt->slist[i];
	for (j = 0; j < curr_structure->num_contours; j++) {
	    Rtss_polyline *curr_polyline = curr_structure->pslist[j];
	    if (curr_polyline->num_vertices <= 0) {
		continue;
	    }
	    gs.get_slice_info (
		&curr_polyline->slice_no,
		&curr_polyline->ct_slice_uid,
		curr_polyline->z[0]);
	}
    }
}
#endif

void
Rtss_polyline_set::set_geometry_from_plm_image_header (
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
Rtss_polyline_set::set_geometry_from_plm_image (
    Plm_image *pli
)
{
    Plm_image_header pih;
    pih.set_from_plm_image (pli);
    this->set_geometry_from_plm_image_header (&pih);
}
