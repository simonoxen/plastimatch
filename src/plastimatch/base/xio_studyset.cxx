/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dir_list.h"
#include "path_util.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "xio_studyset.h"

int Xio_studyset::gcd(int a, int b) {

    // Euclidean algorithm

    while (b != 0) {
	int t = b;
	b = a % b;
	a = t;
    }

    return a;
}

Xio_studyset::Xio_studyset (const std::string& input_dir)
{
    this->studyset_dir = input_dir;

    // Open index.dat file in input_dir
    std::string indexdat = compose_filename (input_dir, "index.dat");
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
        index.close ();
    } else {
        // Older data has no index.dat file.  Use the filenames instead.
        Dir_list d (input_dir);
        for (int i = 0; i < d.num_entries; i++) {
            std::string entry = d.entry(i);
            if (extension_is (entry, "CT")) {
                std::string fn = basename (entry);
                std::string loc = fn.substr (2, fn.length()-4);
                //printf ("loc = %s\n", loc.c_str());
                float slice_location = std::stof (loc);
                //printf ("%f %s\n", slice_location, fn.c_str());
                Xio_studyset_slice slice (fn, slice_location);
                all_slices.push_back (slice);
            }
        }
    }
    
    // Sort slices in positive direction
    std::sort (all_slices.begin(), all_slices.end());

    // Workaround for Xio position rounding.  Xio rounds positions to the
    // nearest 0.1 mm.  This causes the inequal slice spacing workaround
    // to unnecessarily trigger.
    for (auto it = all_slices.begin(); it != all_slices.end(); it++) {
        long this_location = ROUND_INT (10.f * fabs(it->location));
        long this_modulo = this_location % 10;
        printf ("%f", it->location);
        if (this_modulo == 3 || this_modulo == 8)
        {
            if (it->location < 0) {
                it->location += 0.05;
            } else {
                it->location -= 0.05;
            }
        }
        printf (" -> %f\n", it->location);
    }
    
    // Workaround for multiple slice thickness
    // Create volume with uniform voxel sizes by finding greatest
    // common divisor of slice thicknesses,
    // and duplicating slices to obtain a uniform Z axis.
    std::vector<float> slice_thickness;
    for (auto it = all_slices.begin(); it != all_slices.end(); it++) {
        float prev_spacing = FLT_MAX;
        float next_spacing = FLT_MAX;
        if (it != all_slices.begin()) {
            prev_spacing = it->location - prev(it)->location;
            prev_spacing = ROUND_INT (prev_spacing * 100) / 100.f;
        }
        if (next(it) != all_slices.end()) {
            next_spacing = next(it)->location - it->location;
            next_spacing = ROUND_INT (next_spacing * 100) / 100.f;
        }
        printf ("-> %f\n", std::min (prev_spacing, next_spacing));
        slice_thickness.push_back (std::min (prev_spacing, next_spacing));
    }
    
    // Find greatest common divisor
    std::vector<int> slice_thickness_int;
    int slice_thickness_gcd = 1;

    for (size_t i = 0; i < all_slices.size(); i++) {
	// 1/1000 mm resolution
	int rounded_thickness = static_cast<int> (slice_thickness[i] * 1000.);
	if (rounded_thickness == 0) rounded_thickness = 1;
	slice_thickness_int.push_back(rounded_thickness);
    }

    if (all_slices.size() == 1) {
	slice_thickness_gcd = slice_thickness_int[0];
    }
    else if (all_slices.size() > 0) {
	slice_thickness_gcd = gcd(slice_thickness_int[0], slice_thickness_int[1]);
	for (size_t i = 2; i < all_slices.size(); i++) {
            slice_thickness_gcd = gcd(slice_thickness_gcd, slice_thickness_int[i]);
	}
    }

    // Build new slice list, determining duplication needed for each slice
    thickness = slice_thickness_gcd / 1000.;
    number_slices = 0;

    if (all_slices.size() > 0) {
	float location = all_slices[0].location - (slice_thickness[0] / 2.) + (thickness / 2.);

	for (size_t i = 0; i < all_slices.size(); i++) {
	    int duplicate = slice_thickness_int[i] / slice_thickness_gcd;

	    for (int j = 0; j < duplicate; j++) {
                Xio_studyset_slice slice(all_slices[i].filename_scan, location);
                slices.push_back(slice);
                location += thickness;
                number_slices++;
	    }
	}
    }
    
    // Initialize pixel spacing to zero.  This get set when the 
    // CT is loaded
    this->ct_pixel_spacing[0] = this->ct_pixel_spacing[1] = 0.f;
}

Xio_studyset::~Xio_studyset ()
{
}


Xio_studyset_slice::Xio_studyset_slice (std::string slice_filename_scan, const float slice_location)
{
    filename_scan = slice_filename_scan;
    location = slice_location;

    // Get name from slice filename
    size_t extension_dot = filename_scan.find_last_of("."); 
    name = filename_scan.substr(0, extension_dot);

    filename_contours = name + ".WC";
}

Xio_studyset_slice::~Xio_studyset_slice ()
{
}
