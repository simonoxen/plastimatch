/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "proton_dose_opts.h"
#include "math_util.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

void
print_usage (void)
{
    printf (
	"Usage: proton_dose [options] infile outfile\n"
	"Options:\n"
	" -A hardware       Either \"cpu\" or \"cuda\" (default=cpu)\n"
	" -src \"x y z\"      ...\n"
	" -iso \"x y z\"      ...\n"
	" -vup \"x y z\"      ...\n"
	" -s scale          Scale the intensity of the output file\n"
	" -u step           Uniform step (in mm) along ray trace\n"
    " -p filename       Proton dose energy profile\n"
	" --debug           Create various debug files\n"
    );
    exit (1);
}

void
proton_dose_opts_init (Proton_dose_options* options)
{
    options->threading = THREADING_CPU;
    options->src[0] = -2000.0f;
    options->src[1] = 0.0f;
    options->src[2] = 0.0f;
    options->isocenter[0] = 0.0f;
    options->isocenter[1] = 0.0f;
    options->isocenter[2] = 0.0f;
    options->vup[0] = 0.0f;
    options->vup[1] = 0.0f;
    options->vup[2] = 1.0f;

    options->scale = 1.0f;
    options->ray_step = 1.0f;
    options->input_fn = 0;
    options->input_pep_fn = 0;
    options->output_fn = 0;
    options->debug = 0;
}

void
proton_dose_parse_args (Proton_dose_options* options, int argc, char* argv[])
{
    int i, rc;

    proton_dose_opts_init (options);
    for (i = 1; i < argc; i++) {
	//printf ("ARG[%d] = %s\n", i, argv[i]);
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-A")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		    fprintf(stderr, "option %s requires an argument\n", argv[i]);
		    exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "brook") || !strcmp(argv[i], "BROOK")
		|| !strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")
		|| !strcmp(argv[i], "gpu") || !strcmp(argv[i], "GPU"))
	    {
		    options->threading = THREADING_CUDA;
	    }
	    else {
		    options->threading = THREADING_CPU;
	    }
	}
	else if (!strcmp (argv[i], "-src")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f %f", 
		&options->src[0],
		&options->src[1], 
		&options->src[2]);
	    if (rc != 3) {
		    print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-iso")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f %f", 
		&options->isocenter[0],
		&options->isocenter[1], 
		&options->isocenter[2]);
	    if (rc != 3) {
		    print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-vup")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f %f", 
		&options->vup[0],
		&options->vup[1], 
		&options->vup[2]);
	    if (rc != 3) {
		    print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-s")) {
	    i++;
	    rc = sscanf (argv[i], "%g" , &options->scale);
	    if (rc != 1) {
		    print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-u")) {
	    i++;
	    rc = sscanf (argv[i], "%f" , &options->ray_step);
	    if (rc != 1) {
		    print_usage ();
	    }
	}
    else if (!strcmp (argv[i], "-p")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
	    	fprintf(stderr, "option %s requires an argument\n", argv[i]);
		    exit(1);
	    }
	    i++;
	    options->input_pep_fn = strdup (argv[i]);
    }
    else if (!strcmp (argv[i], "--debug")) {
	    options->debug = 1;
	}
	else {
	    print_usage ();
	    break;
	}
    }

    if (i+1 >= argc) {
	print_usage ();
    }

    if (!options->input_pep_fn) {
        printf ("  Must specify a proton energy profile (-p switch)\n");
        printf ("  Terminating...\n\n");
        exit(1);
    }


    options->input_fn = argv[i];
    options->output_fn = argv[i+1];
}
