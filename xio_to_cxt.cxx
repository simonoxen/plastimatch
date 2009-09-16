/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdlib.h>
#include <stdio.h>

#include "plm_config.h"
#include "readcxt.h"
#include "xio_load.h"

void
print_usage (void)
{
    printf ("Usage: cms_to_cxt directory output_file.cxt x_adj y_adj\n");
}

void
do_cms_to_cxt (char *input_dir, char *output_fn, float x_adj, float y_adj)
{
    Cxt_structure_list structures;

    xio_load_structures (&structures, input_dir, x_adj, y_adj);

    /* Write out the cxt */
    cxt_write (&structures, output_fn);
}

int 
main (int argc, char* argv[]) 
{
    char *input_dir, *output_fn;
    float x_adj = 0.0, y_adj = 0.0;

    if (argc != 5) {
	print_usage ();
	return 1;
    }

    input_dir = argv[1];
    output_fn = argv[2];
    x_adj = atof (argv[3]);
    y_adj = atof (argv[4]);

    do_cms_to_cxt (input_dir, output_fn, x_adj, y_adj);

    return 0;
}
