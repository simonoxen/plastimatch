/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "bspline_opts.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

void
print_usage (void)
{
    printf ("Usage: bspline [options] fixed moving\n"
	    "Options:\n"
	    " -A implementation          Either \"cpu\" or \"brook\" or \"cuda\" (default=cpu)\n"
	    " -a { steepest | lbfgsb }   Choose optimization algorithm\n"
	    " -M { mse | mi }            Registration metric (default is mse)\n"
	    " -m iterations              Maximum iterations (default is 10)\n"
	    " -s \"i j k\"                 Integer knot spacing (voxels)\n"
	    " -O outfile                 The output file\n"
	    " --debug                    Create various debug files\n"
	    );
    exit (1);
}

void
parse_args (BSPLINE_Options* options, int argc, char* argv[])
{
    int d, i, rc;
    BSPLINE_Parms* parms = &options->parms;

    options->output_fn = "output.mha";
    for (d = 0; d < 3; d++) {
	options->vox_per_rgn[d] = 15;
    }
    bspline_parms_set_default (parms);

    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-A")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "brook") || !strcmp(argv[i], "BROOK")
		|| !strcmp(argv[i], "gpu") || !strcmp(argv[i], "GPU")) {
		parms->implementation = BIMPL_BROOK;
	    } 
	    else if(!strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")) {
		parms->implementation = BIMPL_CUDA;
	    }
	    else {
		parms->implementation = BIMPL_CPU;
	    }
	}
        else if (!strcmp (argv[i], "-a")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "steepest")) {
		parms->optimization = BOPT_STEEPEST;
	    } else if (!strcmp(argv[i], "lbfgsb")) {
		parms->optimization = BOPT_LBFGSB;
	    } else {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-m")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%d" , &parms->max_its);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-M")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "mse")) {
		parms->metric = BMET_MSE;
	    } else if (!strcmp(argv[i], "mi")) {
		parms->metric = BMET_MI;
	    } else {
		print_usage ();
	    }
	}
        else if (!strcmp (argv[i], "-s")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%d %d %d", 
			 &options->vox_per_rgn[0],
			 &options->vox_per_rgn[1],
			 &options->vox_per_rgn[2]);
	    if (rc == 1) {
		options->vox_per_rgn[1] = options->vox_per_rgn[0];
		options->vox_per_rgn[2] = options->vox_per_rgn[0];
	    } else if (rc != 3) {
		print_usage ();
	    }
	}
        else if (!strcmp (argv[i], "-O")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->output_fn = strdup (argv[i]);
	}
        else if (!strcmp (argv[i], "--debug")) {
	    parms->debug = 1;
	}
	else {
	    print_usage ();
	    break;
	}
    }
    if (i+1 >= argc) {
	print_usage ();
    }
    options->fixed_fn = argv[i];
    options->moving_fn = argv[i+1];
    printf ("Fixed = %s\n", options->fixed_fn);
    printf ("Moving = %s\n", options->moving_fn);
}
