/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_metadata.h"
#include "dcmtk_series.h"
#include "dicom_rt_study.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "print_and_exit.h"
#include "volume.h"

class Dcmtk_series_private {
public:
    std::list<Dcmtk_file*> m_flist;
    Dicom_rt_study *m_drs;

public:
    Dcmtk_series_private () {
        /* Don't create m_drs.  It is set by caller. */
        m_drs = 0;
    }
    ~Dcmtk_series_private () {
        std::list<Dcmtk_file*>::iterator it;
        for (it = m_flist.begin(); it != m_flist.end(); ++it) {
            delete (*it);
        }

        /* Don't delete m_drs.  It belongs to caller. */
    }
};

Dcmtk_series::Dcmtk_series ()
{
    d_ptr = new Dcmtk_series_private;
}

Dcmtk_series::~Dcmtk_series ()
{
    delete d_ptr;
}

const char*
Dcmtk_series::get_cstr (const DcmTagKey& tag_key) const
{
    return d_ptr->m_flist.front()->get_cstr(tag_key);
}

bool
Dcmtk_series::get_int16_array (const DcmTagKey& tag_key, 
    const int16_t** val, unsigned long* count) const
{
    return d_ptr->m_flist.front()->get_int16_array (tag_key, val, count);
}

bool
Dcmtk_series::get_sequence (const DcmTagKey& tag_key,
    DcmSequenceOfItems*& seq) const
{
    return d_ptr->m_flist.front()->get_sequence (tag_key, seq);
}

std::string 
Dcmtk_series::get_string (const DcmTagKey& tag_key) const
{
    const char* c = d_ptr->m_flist.front()->get_cstr(tag_key);
    if (!c) c = "";
    return std::string(c);
}

bool
Dcmtk_series::get_uint16 (const DcmTagKey& tag_key, uint16_t* val) const
{
    return d_ptr->m_flist.front()->get_uint16 (tag_key, val);
}

bool
Dcmtk_series::get_uint16_array (const DcmTagKey& tag_key, 
    const uint16_t** val, unsigned long* count) const
{
    return d_ptr->m_flist.front()->get_uint16_array (tag_key, val, count);
}

std::string 
Dcmtk_series::get_modality (void) const
{
    return get_string (DCM_Modality);
}

std::string 
Dcmtk_series::get_referenced_uid (void) const
{
    bool rc;
    if (this->get_modality() != "RTSTRUCT") {
	return "";
    }

    DcmItem* rfors = 0;
    rc = d_ptr->m_flist.front()->get_dataset()->findAndGetSequenceItem (
	DCM_ReferencedFrameOfReferenceSequence, rfors).good();
    if (!rc) {
	return "";
    }
    printf ("Found DCM_ReferencedFrameOfReferenceSequence!\n");

    DcmItem* rss = 0;
    rc = rfors->findAndGetSequenceItem (
	DCM_RTReferencedStudySequence, rss).good();
    if (!rc) {
	return "";
    }
    printf ("Found DCM_RTReferencedStudySequence!\n");

    return "";
}

void
Dcmtk_series::insert (Dcmtk_file *df)
{
    d_ptr->m_flist.push_back (df);
}

void
Dcmtk_series::sort (void)
{
    d_ptr->m_flist.sort (dcmtk_file_compare_z_position);
}

void
Dcmtk_series::set_rt_study (Dicom_rt_study *drs)
{
    d_ptr->m_drs = drs;
}

Plm_image*
Dcmtk_series::load_plm_image (void)
{
    /* Sort in Z direction */
    this->sort ();

    /* GCS FIX:
       (1) Direction cosines
       (2) Minimum 2 slices
       (3) Consistency of images w/in series
       (4) Rescale offset/slope
       (5) Different image types
       (6) Refine slice spacing based on entire chunk size
    */

    /* Check for minimum 2 slices */
    if (d_ptr->m_flist.size() < 2) {
	return 0;
    }
    
    /* Get first slice */
    std::list<Dcmtk_file*>::iterator it = d_ptr->m_flist.begin();
    Dcmtk_file *df = (*it);
    float z_init, z_prev, z_diff, z_last;
    int slice_no = 0;
    float best_chunk_z_start = z_init = z_prev = df->m_vh.get_origin()[2];
    
    /* Store UIDs */
    if (d_ptr->m_drs) {
        d_ptr->m_drs->set_ct_series_uid (
            df->get_cstr (DCM_SeriesInstanceUID));
        d_ptr->m_drs->set_frame_of_reference_uid (
            df->get_cstr (DCM_FrameOfReferenceUID));
        d_ptr->m_drs->set_study_uid (
            df->get_cstr (DCM_StudyInstanceUID));
        d_ptr->m_drs->set_study_date (
            df->get_cstr (DCM_StudyDate));
        d_ptr->m_drs->set_study_time (
            df->get_cstr (DCM_StudyTime));

        /* Store remaining metadata */
        Metadata *study_metadata = d_ptr->m_drs->get_study_metadata ();
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientName);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientID);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientSex);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientPosition);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_StudyID);

        Metadata *image_metadata = d_ptr->m_drs->get_image_metadata ();
        dcmtk_copy_into_metadata (image_metadata, df, DCM_Modality);
    }

    /* Get next slice */
    ++it; ++slice_no;
    df = (*it);
    z_diff = df->m_vh.get_origin()[2] - z_prev;
    z_last = z_prev = df->m_vh.get_origin()[2];

    /* We want to find the largest chunk with equal spacing.  This will 
       be used to resample in the case of irregular spacing. */
    int this_chunk_start = 0, best_chunk_start = 0;
    float this_chunk_diff = z_diff, best_chunk_diff = z_diff;
    int this_chunk_len = 2, best_chunk_len = 2;

    /* Loop through remaining slices */
    while (++it != d_ptr->m_flist.end())
    {
	++slice_no;
	df = (*it);
	z_diff = df->m_vh.get_origin()[2] - z_prev;
	z_last = z_prev = df->m_vh.get_origin()[2];

	if (fabs (this_chunk_diff - z_diff) > 0.11) {
	    /* Start a new chunk if difference in thickness is 
	       more than 0.1 millimeter */
	    this_chunk_start = slice_no - 1;
	    this_chunk_len = 2;
	    this_chunk_diff = z_diff;
	} else {
	    /* Same thickness, increase size of this chunk */
	    this_chunk_diff = ((this_chunk_len * this_chunk_diff) + z_diff)
		/ (this_chunk_len + 1);
	    this_chunk_len++;

	    /* Check if this chunk is now the best chunk */
	    if (this_chunk_len > best_chunk_len) {
		best_chunk_start = this_chunk_start;
		best_chunk_len = this_chunk_len;
		best_chunk_diff = this_chunk_diff;
		best_chunk_z_start = z_prev 
		    - (best_chunk_len-1) * best_chunk_diff;
	    }
	}
    }

    /* Report information about best chunk */
    printf ("Best chuck:\n  Slices %d to %d from (0 to %d)\n"
	"  Z_loc = %f %f\n" 
	"  Slice spacing = %f\n", 
	best_chunk_start, best_chunk_start + best_chunk_len - 1, slice_no, 
	best_chunk_z_start, 
	best_chunk_z_start + (best_chunk_len - 1) * best_chunk_diff, 
	best_chunk_diff);

    /* Some debugging info */
    printf ("Slices: ");
    for (it = d_ptr->m_flist.begin(); it != d_ptr->m_flist.end(); ++it) {
	Dcmtk_file *df = (*it);
	printf ("%f ", df->m_vh.get_origin()[2]);
    }
    printf ("\n");

    /* Create a Volume_header to hold the image geometry */
    Volume_header vh;
    plm_long *dim = vh.get_dim();

    /* Compute resampled volume header */
    int slices_before = 
	ROUND_INT ((best_chunk_z_start - z_init) / best_chunk_diff);
    int slices_after = 
	ROUND_INT ((z_last - best_chunk_z_start 
		- (best_chunk_len - 1) * best_chunk_diff) / best_chunk_diff);
    df = (*d_ptr->m_flist.begin());
    vh.clone (&df->m_vh);
    dim[2] = slices_before + best_chunk_len + slices_after;
    vh.get_origin()[2] = best_chunk_z_start - slices_before * best_chunk_diff;
    vh.get_spacing()[2] = best_chunk_diff;

    /* Store image header */
    if (d_ptr->m_drs) {
        d_ptr->m_drs->set_image_header (Plm_image_header (vh));
    }

    /* More debugging info */
    vh.print ();

    /* Still more debugging info */
    printf ("Resamples slices: ");
    for (plm_long i = 0; i < dim[2]; i++) {
	printf ("%f ", vh.get_origin()[2] + i * vh.get_spacing()[2]);
    }
    printf ("\n");

    /* Divine image type */
    df = (*d_ptr->m_flist.begin());
    uint16_t samp_per_pix, bits_alloc, bits_stored, high_bit, pixel_rep;
    const char* phot_interp;
    bool rc = df->get_uint16 (DCM_SamplesPerPixel, &samp_per_pix);
    if (!rc) {
	return 0;
    }
    phot_interp = df->get_cstr (DCM_PhotometricInterpretation);
    if (!phot_interp) {
	return 0;
    }
    rc = df->get_uint16 (DCM_BitsAllocated, &bits_alloc);
    if (!rc) {
	return 0;
    }
    rc = df->get_uint16 (DCM_BitsStored, &bits_stored);
    if (!rc) {
	return 0;
    }
    rc = df->get_uint16 (DCM_HighBit, &high_bit);
    if (!rc) {
	return 0;
    }
    rc = df->get_uint16 (DCM_PixelRepresentation, &pixel_rep);
    if (!rc) {
	return 0;
    }
    printf ("Samp_per_pix: %d\n", (int) samp_per_pix);
    printf ("Phot_interp: %s\n", phot_interp);
    printf ("Bits_alloc: %d\n", (int) bits_alloc);
    printf ("Bits_stored: %d\n", (int) bits_stored);
    printf ("High_bit: %d\n", (int) high_bit);
    printf ("Pixel_rep: %d\n", (int) pixel_rep);

    /* Some kinds of images we don't know how to deal with.  
       Don't load these. */
    if (samp_per_pix != 1) {
	return 0;
    }
    if (strcmp (phot_interp, "MONOCHROME2")) {
	return 0;
    }
    if (bits_alloc != 16) {
	return 0;
    }
    if (bits_stored != 16) {
	return 0;
    }
    if (high_bit != 15) {
	return 0;
    }
    if (pixel_rep != 1) {
	return 0;
    }

    printf ("Image looks ok.  Try to load.\n");

    Plm_image *pli = new Plm_image;
    pli->m_type = PLM_IMG_TYPE_GPUIT_SHORT;
    pli->m_original_type = PLM_IMG_TYPE_GPUIT_SHORT;
    pli->m_gpuit = new Volume (vh, PT_SHORT, 1);
    Volume* vol = (Volume*) pli->m_gpuit;
    uint16_t* img = (uint16_t*) vol->img;

    for (plm_long i = 0; i < dim[2]; i++) {
	/* Find the best slice, using nearest neighbor interpolation */
	std::list<Dcmtk_file*>::iterator best_slice_it = d_ptr->m_flist.begin();
	float best_z_dist = FLT_MAX;
	float z_pos = vh.get_origin()[2] + i * vh.get_spacing()[2];
	for (it = d_ptr->m_flist.begin(); it != d_ptr->m_flist.end(); ++it) {
	    float this_z_dist = fabs ((*it)->m_vh.get_origin()[2] - z_pos);
	    if (this_z_dist < best_z_dist) {
		best_z_dist = this_z_dist;
		best_slice_it = it;
	    }
	}

	/* Load the slice image data into volume */
	const uint16_t* pixel_data;
	Dcmtk_file *df = (*best_slice_it);
	unsigned long length;

	printf ("Loading slice z=%f at location z=%f\n",
	    (*best_slice_it)->m_vh.get_origin()[2], z_pos);

	rc = df->get_uint16_array (DCM_PixelData, &pixel_data, &length);
	if (!rc) {
	    print_and_exit ("Oops.  Error reading pixel data.  Punting.\n");
	}
	if (((long) length) != dim[0] * dim[1]) {
	    print_and_exit ("Oops.  Dicom image had wrong length "
		"(%d vs. %d x %d).\n", length, dim[0], dim[1]);
	}
	memcpy (img, pixel_data, length * sizeof(uint16_t));
	img += length;

	/* Store slice UID */
        if (d_ptr->m_drs) {
            d_ptr->m_drs->set_slice_uid (i, df->get_cstr (DCM_SOPInstanceUID));
        }
    }

    return pli;
}

void
Dcmtk_series::debug (void) const
{
    std::list<Dcmtk_file*>::const_iterator it;
    for (it = d_ptr->m_flist.begin(); it != d_ptr->m_flist.end(); ++it) {
	Dcmtk_file *df = (*it);
	df->debug ();
    }
}
