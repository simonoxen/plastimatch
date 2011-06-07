/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "reg_opts.h"
#include "reg.h"

void
print_usage (void)
{
    printf (
    "Usage: reg [options] vector_field\n"
    "Options:\n"
    " -f implementation        Choose implementation (single letter: a, b, c, etc)\n\n"
    );
    exit(1);
}

void
reg_opts_parse_args (Reg_options* options, int argc, char* argv[])
{
    int i;
    Reg_parms* parms = &options->parms;

    for (i=1; i < argc; i++) {
    if (argv[i][0] != '-') break;
    if (!strcmp (argv[i], "-f")) {
        if (i == (argc-1) || argv[i+1][0] == '-') {
        fprintf(stderr, "option %s requires an argument\n", argv[i]);
        exit(1);
        }
        i++;
        parms->implementation = argv[i][0];
    }
    else {
        print_usage ();
        break;
    }
    } /* for (i < argc) */

    /* load input vector field */
    if (i >= argc) {
    	print_usage ();
    }

    options->input_vf_fn = argv[i];
    printf ("Vector Field = %s\n", options->input_vf_fn);

}
