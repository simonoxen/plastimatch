/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
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

#include "plm_config.h"
#include "cxt_io.h"

/* Modified from ITK source code, function RegularExpressionSeriesFileNames::
   GetFileNames() */
struct lt_pair_numeric_string_string
{
  bool operator()(const std::pair<std::string, std::string> s1, 
                  const std::pair<std::string, std::string> s2) const
    {
	return atof(s1.second.c_str()) < atof(s2.second.c_str());
    }
};

struct lt_pair_alphabetic_string_string
{
  bool operator()(const std::pair<std::string, std::string> s1, 
                  const std::pair<std::string, std::string> s2) const
    {
	return s1.second < s2.second;
    }
};

static void
get_file_names (std::vector<std::pair<std::string,std::string> > *file_names,
		const char *input_dir, const char *regular_expression)
{
    int m_SubMatch = 1;
    int m_NumericSort = 1;
    std::string m_Directory = input_dir;
    itksys::RegularExpression reg;

    if (!reg.compile(regular_expression)) {
	printf ("Error\n");exit (-1);
    }

    // Process all files in the directory
    itksys::Directory fileDir;
    if (!fileDir.Load (input_dir)) {
	printf ("Error\n");exit (-1);
    }

    // Scan directory for files. Each file is checked to see if it
    // matches the m_RegularExpression.
    for (unsigned long i = 0; i < fileDir.GetNumberOfFiles(); i++) {
	// Only read files
	if (itksys::SystemTools::FileIsDirectory ((m_Directory + "/" + fileDir.GetFile(i)).c_str())) {
	    continue;
	}

	if (reg.find (fileDir.GetFile(i))) {
	    // Store the full filename and the selected sub expression match
	    std::pair<std::string,std::string> fileNameMatch;
	    fileNameMatch.first = m_Directory + "/" + fileDir.GetFile(i);
	    fileNameMatch.second = reg.match(m_SubMatch);
	    file_names->push_back(fileNameMatch);
	}
    }
  
    // Sort the files. The files are sorted by the sub match defined by
    // m_SubMatch. Sorting can be alpahbetic or numeric.
    if (m_NumericSort)
    {
	std::sort(file_names->begin(),
		  file_names->end(),
		  lt_pair_numeric_string_string());
    }
    else
    {
	std::sort(file_names->begin(),
		  file_names->end(),
		  lt_pair_alphabetic_string_string());
    }
}

static void
add_cms_contournames (Cxt_structure_list *structures, const char *filename)
{
    FILE *fp;
    struct bStream * bs;
    bstring line1 = bfromcstr ("");
    bstring line2 = bfromcstr ("");

    fp = fopen (filename, "r");
    if (!fp) {
	printf ("Error opening file %s for read\n", filename);
	exit (-1);
    }

    bs = bsopen ((bNread) fread, fp);
    printf ("0 [%d,%d]> %s\n", line1->mlen, line1->slen, line1->data);

    bsreadln (line1, bs, '\n');
    bsreadln (line1, bs, '\n');

    while (1)
    {
	int rc, id;

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
	rc = sscanf ((char*) line2->data, "%d,", &id);
	if (rc != 1) {
	    fprintf (stderr, "Error parsing contournames: contour id not found (%s)\n", line1->data);
	    exit (-1);
	}

	/* Add structure */
	cxt_add_structure (structures, (char*) line1->data, id);

	/* Skip 2 lines */
	bsreadln (line1, bs, '\n');
	bsreadln (line1, bs, '\n');
    }

    cxt_debug_structures (structures);

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
	    printf ("Error parsing file %s (num_points)\n", filename);
	    exit (-1);
	}

	/* Get structure number */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &structure_id);
	if (rc != 1) {
	    printf ("Error parsing file %s (structure_id)\n", filename);
	    exit (-1);
	}

	/* Can this happen? */
	if (num_points == 0) {
	    break;
	}

	/* Look up the cxt structure for this id */
	curr_structure = cxt_find_structure_by_id (structures, structure_id);
	if (!curr_structure) {
	    printf ("Couldn't reference structure with id %d\n", structure_id);
	    exit (-1);
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
		    printf ("Error parsing file %s (points) %s\n", filename, &buf[line_loc]);
		    exit (-1);
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

void
xio_load_structures (Cxt_structure_list *structures, char *input_dir, 
		     float x_adj, float y_adj)
{
    
    const char *filename_re = "T\\.([-\\.0-9]*)\\.WC";

    /* Get the index file */
    std::string index_file = std::string(input_dir) + "/" + "contournames";
    if (!itksys::SystemTools::FileExists (index_file.c_str(), true)) {
	fprintf (stderr, "No xio contournames file found in directory %s\n", input_dir);
	exit (-1);
    }

    /* Get the list of filenames */
    std::vector<std::pair<std::string,std::string> > file_names;
    get_file_names (&file_names, input_dir, filename_re);
    if (file_names.empty ()) {
	fprintf (stderr, "No xio structure files found in directory %s\n", input_dir);
	exit (-1);
    }

    /* Load the index file */
    cxt_initialize (structures);
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
}

