/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "xio_studyset.h"

Xio_studyset::Xio_studyset (const char *input_dir)
{
    this->studyset_dir = input_dir;

    // Open index.dat file in input_dir
    std::string indexdat(input_dir);
    indexdat += "/index.dat";
    std::ifstream index (indexdat.c_str());

    int all_number_slices = 0;
    std::vector<Xio_studyset_slice> all_slices;

    if (index.is_open()) {
	// Get total number of slices
	index >> all_number_slices;

	// Loop through slices getting filename and location
	std::string slice_filename_scan;
	std::string slice_name;
	float slice_location;

	for (int i = 0; i < all_number_slices; i++) {
	    index >> slice_filename_scan;
	    index >> slice_location;

	    Xio_studyset_slice slice (slice_filename_scan, slice_location);
	    all_slices.push_back (slice);
	}

	// Sort slices in positive direction
	std::sort (all_slices.begin(), all_slices.end());
    } else {
	all_number_slices = 0;
    }

    // Plastimatch only supports volumes with uniform voxel sizes
    // If slice thickness is not uniform, extract the largest uniform chunk

    float best_chunk_diff;
    int best_chunk_start;
    int best_chunk_len;

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
}

Xio_studyset::~Xio_studyset ()
{
}


Xio_studyset_slice::Xio_studyset_slice (std::string slice_filename_scan, const float slice_location)
{
    filename_scan = slice_filename_scan;
    location = slice_location;

    // Get name from slice filename
    int extension_dot = filename_scan.find_last_of("."); 
    name = filename_scan.substr(0, extension_dot);

    filename_contours = name + ".WC";
}

Xio_studyset_slice::~Xio_studyset_slice ()
{
}
