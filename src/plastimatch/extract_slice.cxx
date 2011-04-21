/*
   This utility function extract specified slices in CT image.   
 */

#include "plm_config.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "thumbnail.h"
#include "getopt.h"
#include "itkImageRegionIterator.h"
#include <time.h>

#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
using namespace std;

void load_loc_file(const char* in_fn, vector<float> &slice_locs)
{
  int idx, i;
  float tx, ty, tz;

  ifstream ifp (in_fn);
  char cur_line[256];

  // parse the fiducial list file
  if (ifp.is_open()) {
    while (ifp.good()) {
      ifp.getline(cur_line, 256);
      cout << cur_line << endl;
      if (cur_line[0] != '#') {
	// read in the location
	if (sscanf(cur_line, "%d,%f,%f,%f,%d,%d", 
		   &idx, &tx, &ty, &tz, &i, &i) > 0){
	  cout << "tz = " << tz << endl;
	  slice_locs.push_back(tz);
	}
      }
    }
  }
  else {
    cerr << "Cannot open input file " << in_fn << endl;
  }
}


void print_usage()
{
  fprintf(stderr, 
	  "Usage: extract_slice input-image-file slice-loc-file output-image-file-prefix\n");
}

int main (int argc, char *argv[])
{
  Plm_image *pli;
  int num_slices;
  int dim[3]; 
  float spacing[3];

  vector<float> slice_locs;

  string img_in_fn;
  string img_out_fn_prefix;
  string img_out_fn;
  string slice_loc_fn;
    
  
  if (argc < 4) {
    print_usage();
    return -1;
  }

  img_in_fn = argv[1];
  slice_loc_fn = argv[2];
  img_out_fn_prefix = argv[3];

  // load input image
  pli = plm_image_load((const char*) img_in_fn.c_str(), PLM_IMG_TYPE_ITK_FLOAT);
  Plm_image_header pih (pli);
  pih.get_dim(dim);
  pih.get_spacing(spacing);

  // load slice location file
  load_loc_file((const char*) slice_loc_fn.c_str(), slice_locs);

  num_slices = slice_locs.size();
  //cout << num_slices << endl;
  if (num_slices <= 0){
    cerr << "Error with the file of slice locations." << endl;
    return -1;
  }

  // extract slices
  Thumbnail curslice;
  curslice.set_input_image(pli);
  curslice.set_thumbnail_dim(dim[0]);
  curslice.set_thumbnail_spacing(spacing[0]);

  Plm_image *slice = new Plm_image;
  slice->m_original_type = pli->m_original_type;
  slice->m_type = pli->m_type;

  for (int i_slice = 0; i_slice < num_slices; i_slice ++) {
    curslice.set_slice_loc (slice_locs[i_slice]);
    slice->m_itk_float = curslice.make_thumbnail();
    ostringstream sin;
    sin << i_slice+1;
    img_out_fn = img_out_fn_prefix + "_P" + sin.str() + ".mhd";
    slice->save_image((const char*) img_out_fn.c_str());
  }

  return 0;
}
