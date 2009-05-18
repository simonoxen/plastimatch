/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include "itkDirectory.h"

void
print_usage (void)
{
    printf ("Usage: cms_to_cxt directory output_file.cxt\n");
}

int 
main (int argc, char* argv[]) 
{

    if (argc != 3) {
	print_usage ();
	return 1;
    }

    return 0;
}
