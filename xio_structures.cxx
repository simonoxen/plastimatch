/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <vector>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"
#include "bstrlib.h"

#include "cxt_io.h"
#include "file_util.h"
#include "math_util.h"
#include "plm_path.h"
#include "print_and_exit.h"
#include "xio_io.h"
#include "xio_structures.h"

static void
add_cms_contournames (Cxt_structure_list *structures, const char *filename)
{
    FILE *fp;
    struct bStream * bs;
    bstring line1 = bfromcstr ("");
    bstring line2 = bfromcstr ("");

    fp = fopen (filename, "r");
    if (!fp) {
	print_and_exit ("Error opening file %s for read\n", filename);
    }

    bs = bsopen ((bNread) fread, fp);

    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');

    while (1)
    {
	int rc, structure_id;

	/* Get structure name */
	rc = bsreadln (line1, bs, '\n');
	if (rc == BSTR_ERR) {
	    break;
	}
	btrimws (line1);

	/* Get structure number */
	rc = bsreadln (line2, bs, '\n');
	if (rc == BSTR_ERR) {
	    break;
	}
	rc = sscanf ((char*) line2->data, "%d,", &structure_id);
	if (rc != 1) {
	    print_and_exit ("Error parsing contournames: "
			    "contour id not found (%s)\n", line1->data);
	}

	/* Xio structures can be zero.  This is possibly not tolerated 
	   by dicom.  So we modify before inserting into the cxt. */
	structure_id ++;
	if (structure_id <= 0) {
	    print_and_exit ("Error, structure_id was less than zero\n");
	}

	/* Add structure */
	cxt_add_structure (structures, (char*) line1->data, 0, structure_id);

	/* Skip 2 lines */
	bsreadln (line1, bs, '\n');
	bsreadln (line1, bs, '\n');
    }

    bdestroy (line1);
    bdestroy (line2);
    bsclose (bs);
    fclose (fp);
}

static void
add_cms_structure (Cxt_structure_list *structures, const char *filename, 
		   float z_loc, float x_adj, float y_adj)
{
    FILE *fp;
    char buf[1024];

    fp = fopen (filename, "r");
    if (!fp) {
	printf ("Error opening file %s for read\n", filename);
	exit (-1);
    }

    /* Skip first five lines */
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);
    fgets (buf, 1024, fp);

    while (1) {
	int rc;
	int structure_id, num_points;
	int point_idx, remaining_points;
	Cxt_structure *curr_structure;
	Cxt_polyline *curr_polyline;

	/* Get num points */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &num_points);
	if (rc != 1) {
	    print_and_exit ("Error parsing file %s (num_points)\n", filename);
	}

	/* Get structure number */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &structure_id);
	if (rc != 1) {
	    print_and_exit ("Error parsing file %s (structure_id)\n", 
			    filename);
	}
	
	/* Xio structures can be zero.  This is possibly not tolerated 
	   by dicom.  So we modify before inserting into the cxt. */
	structure_id ++;
	if (structure_id <= 0) {
	    print_and_exit ("Error, structure_id was less than zero\n");
	}

	/* Can this happen? */
	if (num_points == 0) {
	    break;
	}

	/* Look up the cxt structure for this id */
	curr_structure = cxt_find_structure_by_id (structures, structure_id);
	if (!curr_structure) {
	    print_and_exit ("Couldn't reference structure with id %d\n", 
			    structure_id);
	}

	printf ("[%f %d %d]\n", z_loc, structure_id, num_points);
	curr_polyline = cxt_add_polyline (curr_structure);
	curr_polyline->slice_no = -1;
	curr_polyline->num_vertices = num_points;
	curr_polyline->x = (float*) malloc (num_points * sizeof(float));
	curr_polyline->y = (float*) malloc (num_points * sizeof(float));
	curr_polyline->z = (float*) malloc (num_points * sizeof(float));

	point_idx = 0;
	remaining_points = num_points;
	while (remaining_points > 0) {
	    int p, line_points, line_loc;

	    fgets (buf, 1024, fp);

	    if (remaining_points > 5) {
		line_points = 5;
	    } else {
		line_points = remaining_points;
	    }
	    line_loc = 0;

	    for (p = 0; p < line_points; p++) {
		float x, y;
		int rc, this_loc;

		rc = sscanf (&buf[line_loc], "%f, %f,%n", &x, &y, &this_loc);
		if (rc != 2) {
		    print_and_exit ("Error parsing file %s (points) %s\n", 
				    filename, &buf[line_loc]);
		}
		curr_polyline->x[point_idx] = x + x_adj;
		curr_polyline->y[point_idx] = - y + y_adj;
		curr_polyline->z[point_idx] = z_loc;
		point_idx ++;
		line_loc += this_loc;
	    }
	    remaining_points -= line_points;
	}
    }

    fclose (fp);
}

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
xio_structures_load (
    Cxt_structure_list *structures, 
    char *input_dir, 
    float x_adj,
    float y_adj
)
{
    
    const char *filename_re = "T\\.([-\\.0-9]*)\\.WC";

    /* Get the index file */
    std::string index_file = std::string(input_dir) + "/" + "contournames";
    if (!itksys::SystemTools::FileExists (index_file.c_str(), true)) {
	print_and_exit ("No xio contournames file found in directory %s\n", 
			input_dir);
    }

    /* Get the list of filenames */
    std::vector<std::pair<std::string,std::string> > file_names;
    xio_io_get_file_names (&file_names, input_dir, filename_re);
    if (file_names.empty ()) {
	print_and_exit ("No xio structure files found in directory %s\n", 
			input_dir);
    }

    /* Load the index file */
    cxt_init (structures);
    add_cms_contournames (structures, index_file.c_str());

    /* Iterate through filenames, adding data to CXT */
    std::vector<std::pair<std::string,std::string> >::iterator it;
    it = file_names.begin();
    while (it != file_names.end()) {
	const char *filename = (*it).first.c_str();
	float z_loc = atof ((*it).second.c_str());
	printf ("File: %s, Loc: %f\n", filename, z_loc);
	add_cms_structure (structures, filename, z_loc, x_adj, y_adj);
	++it;
    }

    cxt_debug (structures);
}

void
xio_structures_save (
    Cxt_structure_list *cxt, 
    char *output_dir
)
{
    FILE *fp;
    int i, j, k, z;
    char fn[_MAX_PATH];

    if (!cxt->have_geometry) {
	print_and_exit ("Sorry, can't output xio format without ct geometry\n");
    }

    /* Write contournames */
    sprintf (fn, "%s/%s", output_dir, "contournames");
    make_directory_recursive (fn);
    fp = fopen (fn, "w");
    if (!fp) {
	print_and_exit ("Error opening output file %s\n", fn);
    }
    fprintf (fp, "00041027\n%d\n", cxt->num_structures);
    for (i = 0; i < cxt->num_structures; i++) {
	Cxt_structure *curr_structure = &cxt->slist[i];
	fprintf (fp, "%s\n", curr_structure->name);
	fprintf (fp, "%d,1.000000,0,1,19691231.190000\n", i);
	fprintf (fp, "General\n");
	fprintf (fp, "1,5,-1,1,0,0\n");
    }
    fclose (fp);

    /* Write WC files */
    for (z = 0; z < cxt->dim[2]; z++) {
	char fn[_MAX_PATH];
	float z_loc = cxt->offset[2] + z * cxt->spacing[2];
	sprintf (fn, "%s/T.%.1f.WC", output_dir, (ROUND (z_loc * 10) / 10.f));
	fp = fopen (fn, "w");
	if (!fp) {
	    print_and_exit ("Error opening output file %s\n", fn);
	}
	fprintf (fp, "00061013\n\n");
	fprintf (fp, "0\n0.000,0.000,0.000\n");
	/* GCS FIX: These seem to be min/max */
	fprintf (fp, "-158.1,-135.6, 147.7,  81.6\n");
	for (i = 0; i < cxt->num_structures; i++) {
	    Cxt_structure *curr_structure = &cxt->slist[i];
	    for (j = 0; j < curr_structure->num_contours; j++) {
		Cxt_polyline *curr_polyline = &curr_structure->pslist[j];
		fprintf (fp, "%d\n", curr_polyline->num_vertices);
		fprintf (fp, "%d\n", i);
		for (k = 0; k < curr_polyline->num_vertices; k++) {
		    fprintf (fp, "%6.1f,%6.1f", 
			curr_polyline->x[k], curr_polyline->y[k]);
		    if (k % 5 == 0) {
			fprintf (fp, "\n");
		    }
		    else if (k < curr_polyline->num_vertices - 1) {
			fprintf (fp, ",");
		    } else {
			fprintf (fp, "\n");
		    }
		}
	    }
	}
	fprintf (fp, "0\n0\n0\nBart\n");
	fclose (fp);
    }
}
