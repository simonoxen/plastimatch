/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include "itkDirectory.h"
#include "itkRegularExpressionSeriesFileNames.h"

void
print_usage (void)
{
    printf ("Usage: cms_to_cxt directory output_file.cxt\n");
}

void
do_cms_to_cxt (char *input_dir, char *output_fn)
{
    typedef itk::RegularExpressionSeriesFileNames NameGeneratorType;
    NameGeneratorType::Pointer name_generator = NameGeneratorType::New ();

    /* Test make sure it is a directory */
    itk::Directory::Pointer itk_dir = itk::Directory::New ();
    if (!itk_dir->Load (input_dir)) {
	printf ("Could not open input directory: %s\n", input_dir);
	exit (-1);
    }

    /* Get file names matching pattern -- they won't be sorted properly though */
    name_generator->SetRegularExpression ("T\\.([-0-9]*)\\.WC");
    name_generator->SetSubMatch (0);
    name_generator->SetDirectory (input_dir);
    std::vector<std::string> input_files;
    input_files = name_generator->GetFileNames ();

    
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
