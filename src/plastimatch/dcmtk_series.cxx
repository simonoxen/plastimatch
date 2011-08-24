/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_series.h"
#include "plm_image.h"
#include "print_and_exit.h"
#include "volume_header.h"

Dcmtk_series::Dcmtk_series ()
{
}

Dcmtk_series::~Dcmtk_series ()
{
    std::list<Dcmtk_file*>::iterator it;
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	delete (*it);
    }
}

void
Dcmtk_series::debug (void) const
{
    std::list<Dcmtk_file*>::const_iterator it;
    for (it = m_flist.begin(); it != m_flist.end(); ++it) {
	Dcmtk_file *df = (*it);
	df->debug ();
    }
}

std::string 
Dcmtk_series::get_modality (void) const
{
    const char* c = m_flist.front()->get_cstr (DCM_Modality);
    if (!c) c = "";
    return std::string(c);
}

void
Dcmtk_series::insert (Dcmtk_file *df)
{
    m_flist.push_back (df);
}

Plm_image*
Dcmtk_series::load_plm_image (void)
{
    Plm_image *pli = new Plm_image;
    Volume_header vh;

    /* Sort in Z direction */
    this->sort ();

    /* Get first slice */
    std::list<Dcmtk_file*>::iterator it;
    it = m_flist.begin();
    Dcmtk_file *df = (*it);
    float z_prev, z_diff;
    z_prev = df->m_vh.m_origin[2];

    /* Get next slice */
    ++it;
    df = (*it);
    z_diff = df->m_vh.m_origin[2] - z_prev;
    z_prev = df->m_vh.m_origin[2];
    printf ("%f\n", z_diff);

    /* Loop through remaining slices */
    while (++it != m_flist.end())
    {
	df = (*it);
	z_diff = df->m_vh.m_origin[2] - z_prev;
	z_prev = df->m_vh.m_origin[2];
	printf ("%f\n", z_diff);
    }

#if defined (commentout)
    /* Try to assess best Z spacing */
    float best_chunk_start = 0;
    float best_chunk_diff;
    int best_chunk_start;
    int best_chunk_len;
    float z_diff;
    float this_chunk_diff;
    int this_chunk_start;
    int this_chunk_len;


    if (all_number_slices > 1) {

	float z_diff;
	float this_chunk_diff;
	int this_chunk_start;
	int this_chunk_len;

	for (int i = 1; i < all_number_slices; i++) {
	    z_diff = all_slices[i].location - all_slices[i-1].location;

	    if (i == 1) {
		// First chunk
		this_chunk_start = best_chunk_start = 0;
		this_chunk_diff = best_chunk_diff = z_diff;
		this_chunk_len = best_chunk_len = 2;
	    } else if (fabs (this_chunk_diff - z_diff) > 0.11) {
		// Start a new chunk if difference in thickness is more than 0.1 millimeter
		this_chunk_start = i - 1;
		this_chunk_len = 2;
		this_chunk_diff = z_diff;
	    } else {
		// Same thickness, increase size of this chunk
		this_chunk_diff = ((this_chunk_len * this_chunk_diff) + z_diff)
		    / (this_chunk_len + 1);
		this_chunk_len++;

		// Check if this chunk is now the best chunk
		if (this_chunk_len > best_chunk_len) {
		    best_chunk_start = this_chunk_start;
		    best_chunk_len = this_chunk_len;
		    best_chunk_diff = this_chunk_diff;
		}
	    }
	}
    } else {
	best_chunk_start = 0;
	best_chunk_len = 1;
	best_chunk_diff = 0;
    }

    // Extract best chunk
    number_slices = best_chunk_len;
    thickness = best_chunk_diff;
    for (int i = 0; i < best_chunk_len; i++) {
	slices.push_back(all_slices[best_chunk_start + i]);
    }
#endif



    return pli;
}

void
Dcmtk_series::sort (void)
{
    m_flist.sort (dcmtk_file_compare_z_position);
}
