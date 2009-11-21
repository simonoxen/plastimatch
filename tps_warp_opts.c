/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tps_warp_opts.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

void
print_usage (void)
{
    printf ("Usage: tps_warp [options] tps_xform moving\n"
	    "Options:\n"
	    " -V outfile                 The output vector field\n"
	    " -O outfile                 The output warped image\n"
	    );
    exit (1);
}

void
tps_warp_opts_parse_args (Tps_options* options, int argc, char* argv[])
{
    int i;
    memset (options, 0, sizeof (Tps_options));

    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
        if (!strcmp (argv[i], "-O")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->output_warped_fn = strdup (argv[i]);
	}
        else if (!strcmp (argv[i], "-V")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->output_vf_fn = strdup (argv[i]);
	}
	else {
	    print_usage ();
	    break;
	}
    }
    if (i+1 >= argc) {
	print_usage ();
    }
    options->tps_xf_fn = argv[i];
    options->moving_fn = argv[i+1];
    printf ("TPS XF = %s\n", options->tps_xf_fn);
    printf ("Moving = %s\n", options->moving_fn);
}
