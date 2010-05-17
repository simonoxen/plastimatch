 /* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>

#include <itksys/SystemTools.hxx>
#include <itksys/Directory.hxx>
#include <itksys/RegularExpression.hxx>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"
#include "bstrlib.h"

#include "print_and_exit.h"
#include "xio_io.h"

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
xio_io_get_file_names (
    std::vector<std::pair<std::string,std::string> > *file_names,
    const char *input_dir, 
    const char *regular_expression
)
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
