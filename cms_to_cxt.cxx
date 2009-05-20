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

#include "plm_config.h"
#include "readcxt.h"

void
print_usage (void)
{
    printf ("Usage: cms_to_cxt directory output_file.cxt\n");
}

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

void
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

    //    std::vector<std::pair<std::string,std::string> > sorted_filenames;

    // Scan directory for files. Each file is checked to see if it
    // matches the m_RegularExpression.
    for (unsigned long i = 0; i < fileDir.GetNumberOfFiles(); i++)
    {
	// Only read files
	if (itksys::SystemTools::FileIsDirectory( (m_Directory + "/" + fileDir.GetFile(i)).c_str() ))
	{
	    continue;
	}

	if (reg.find(fileDir.GetFile(i)))
	{
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

void
add_cms_structure (STRUCTURE_List *structures, const char *filename, float z_loc)
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
	int structure_number, num_points;
	int remaining_points;

	/* Get num points */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &num_points);
	if (rc != 1) {
	    printf ("Error parsing file %s (num_points)\n", filename);
	    exit (-1);
	}

	/* Get structure number */
	fgets (buf, 1024, fp);
	rc = sscanf (buf, "%d", &structure_number);
	if (rc != 1) {
	    printf ("Error parsing file %s (structure_number)\n", filename);
	    exit (-1);
	}

	if (num_points == 0) {
	    break;
	}

	remaining_points = num_points;
	while (remaining_points > 0) {
	    int p, line_points, idx;

	    fgets (buf, 1024, fp);

	    if (remaining_points > 5) {
		line_points = 5;
	    } else {
		line_points = remaining_points;
	    }
	    idx = 0;

	    for (p = 0; p < line_points; p++) {
		float x, y;
		int rc, this_idx;

		rc = sscanf (&buf[idx], "%f, %f,%n", &x, &y, &this_idx);
		if (rc != 2) {
		    printf ("Error parsing file %s (points) %s\n", filename, &buf[idx]);
		    exit (-1);
		}
		idx += this_idx;
	    }
	    remaining_points -= line_points;
	}
    }

    fclose (fp);
}

void
do_cms_to_cxt (char *input_dir, char *output_fn)
{
    STRUCTURE_List structures;
    const char *filename_re = "T\\.([-\\.0-9]*)\\.WC";

    /* Get the list of filenames */
    std::vector<std::pair<std::string,std::string> > file_names;
    get_file_names (&file_names, input_dir, filename_re);

    /* Iterate through filenames, adding data to CXT */
    cxt_initialize (&structures);
    std::vector<std::pair<std::string,std::string> >::iterator it;
    it = file_names.begin();
    while (it != file_names.end()) {
	const char *filename = (*it).first.c_str();
	float z_loc = atof ((*it).second.c_str());
	printf ("File: %s, Loc: %f\n", filename, z_loc);
	add_cms_structure (&structures, filename, z_loc);
	++it;
    }
}

int 
main (int argc, char* argv[]) 
{
    char *input_dir, *output_fn;
    if (argc != 3) {
	print_usage ();
	return 1;
    }

    input_dir = argv[1];
    output_fn = argv[2];

    do_cms_to_cxt (input_dir, output_fn);

    return 0;
}
