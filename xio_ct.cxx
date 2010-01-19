 /* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <algorithm>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"
#include "bstrlib.h"

#include "cxt_io.h"
#include "print_and_exit.h"
#include "xio_ct.h"
#include "xio_io.h"
#include "xio_structures.h"

/* The x_adj, and y_adj are currently done manually, until I get experience 
   to do automatically.  Here is how I think it is done:
   
   1) Open any of the .CT files
   2) Look for the lines like this:

        0
        230.000,230.000
        512,512,16

   3) The (230,230) values are the location of the isocenter within the 
      slice relative to the upper left pixel.  
   4) The cxt file will normally get assigned an OFFSET field based 
      on the ImagePatientPosition from the dicom set, such as:

        OFFSET -231.6 -230 -184.5

   5) So, in the above example, we should set --x-adj=-1.6, to translate 
      the structures from XiO coordinates to Dicom.
*/
void
xio_ct_load (PlmImage *plm, char *input_dir)
{
    const char *filename_re = "T\\.([-\\.0-9]*)\\.CT";

    /* Get the list of filenames */
    std::vector<std::pair<std::string,std::string> > file_names;
    xio_io_get_file_names (&file_names, input_dir, filename_re);
    if (file_names.empty ()) {
	print_and_exit ("No xio structure files found in directory %s\n", 
			input_dir);
    }

    /* Iterate through filenames, adding data to plm */
    std::vector<std::pair<std::string,std::string> >::iterator it;
    it = file_names.begin();
    while (it != file_names.end()) {
	const char *filename = (*it).first.c_str();
	float z_loc = atof ((*it).second.c_str());
	printf ("File: %s, Loc: %f\n", filename, z_loc);
	//add_cms_structure (structures, filename, z_loc, x_adj, y_adj);
	++it;
    }
}
