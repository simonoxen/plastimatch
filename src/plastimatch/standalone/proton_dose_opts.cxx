/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "proton_dose.h"
#include "plm_math.h"

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
	" -f implementation One of {a,b,c}\n"
	" -src \"x y z\"      ...\n"
	" -iso \"x y z\"      ...\n"
	" -vup \"x y z\"      ...\n"
	" -s scale          Scale the intensity of the output file\n"
	" -d detail         0 = full, 1 = beam path only (default=0)\n"
	" -u step           Uniform step (in mm) along ray trace\n"
	" -p filename       Proton dose energy profile\n"
	" --debug           Create various debug files\n"
    );
    exit (1);
}

void
proton_dose_parms_init (Proton_dose_parms* parms)
{
    parms->threading = THREADING_CPU_OPENMP;
    parms->flavor = 'a';
    parms->src[0] = -2000.0f;
    parms->src[1] = 0.0f;
    parms->src[2] = 0.0f;
    parms->isocenter[0] = 0.0f;
    parms->isocenter[1] = 0.0f;
    parms->isocenter[2] = 0.0f;
    parms->vup[0] = 0.0f;
    parms->vup[1] = 0.0f;
    parms->vup[2] = 1.0f;

    parms->scale = 1.0f;
    parms->ray_step = 1.0f;
    parms->input_fn = 0;
    parms->input_pep_fn = 0;
    parms->output_fn = 0;
    parms->debug = 0;

    parms->detail = 0;
}

void
proton_dose_parse_args (Proton_dose_parms* parms, int argc, char* argv[])
{
    int i, rc;

    proton_dose_parms_init (parms);
    for (i = 1; i < argc; i++) {
	//printf ("ARG[%d] = %s\n", i, argv[i]);
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-A")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")
		|| !strcmp(argv[i], "gpu") || !strcmp(argv[i], "GPU"))
	    {
		parms->threading = THREADING_CUDA;
	    }
	    else {
		parms->threading = THREADING_CPU_OPENMP;
	    }
	}
	else if (!strcmp (argv[i], "-src")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f %f", 
		&parms->src[0],
		&parms->src[1], 
		&parms->src[2]);
	    if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-iso")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f %f", 
		&parms->isocenter[0],
		&parms->isocenter[1], 
		&parms->isocenter[2]);
	    if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-vup")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f %f", 
		&parms->vup[0],
		&parms->vup[1], 
		&parms->vup[2]);
	    if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-d")) {
	    i++;
	    rc = sscanf (argv[i], "%i" , &parms->detail);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-f")) {
	    i++;
	    rc = sscanf (argv[i], "%c" , &parms->flavor);
	    if (rc != 1 || parms->flavor < 'a' || parms->flavor > 'z') {
		fprintf (stderr, 
		    "option %s must be a character beween 'a' and 'z'\n", 
		    argv[i-1]);
		exit (1);
	    }
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
	    parms->input_pep_fn = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-s")) {
	    i++;
	    rc = sscanf (argv[i], "%g" , &parms->scale);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-u")) {
	    i++;
	    rc = sscanf (argv[i], "%f" , &parms->ray_step);
	    if (rc != 1) {
		print_usage ();
	    }
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

    if (!parms->input_pep_fn) {
        printf ("  Must specify a proton energy profile (-p switch)\n");
        printf ("  Terminating...\n\n");
        exit(1);
    }

    parms->input_fn = argv[i];
    parms->output_fn = argv[i+1];
}
