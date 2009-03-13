/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "fdk_opts.h"
#include "volume.h"

void print_usage (void)
{
    printf ("Usage: mghcbct [options]\n"
	    "Options:\n"
	    " -a \"num ((num) num)\"   Use this range of images\n"
	    " -r \"r1 r2 r3\"          Set output resolution (in voxels)\n"
	    " -s scale               Scale the intensity of the output file\n"
	    " -z \"s1 s2 s3\"          Physical size of the reconstruction (in mm)\n"
	    " -I indir               The input directory\n"
	    " -O outfile             The output file\n"
	    );
    exit (1);
}

void set_default_options (MGHCBCT_Options* options)
{
    options->first_img = 0;
    options->last_img = 119;
    options->resolution[0] = 120;
    options->resolution[1] = 120;
    options->resolution[2] = 120;
    options->vol_size[0] = 500.0f;
    options->vol_size[1] = 500.0f;
    options->vol_size[2] = 500.0f;
    options->scale = 1.0f;
    options->input_dir = ".";
    options->output_file = "output.mha";
}

void parse_args (MGHCBCT_Options* options, int argc, char* argv[])
{
    int i, rc;
	
    if (argc < 2)
	{ print_usage(); exit(1); }

    set_default_options (options);
    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-r")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%d %d %d", 
			 &options->resolution[0], 
			 &options->resolution[1],
			 &options->resolution[2]);
	    if (rc == 1) {
		options->resolution[1] = options->resolution[0];
		options->resolution[2] = options->resolution[0];
	    } else if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-I")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->input_dir = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-O")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->output_file = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-a")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%d %d %d" , 
			 &options->first_img,
			 &options->skip_img,
			 &options->last_img);
	    if (rc == 1) {
		options->last_img = options->first_img;
		options->skip_img = 1;
	    } else if (rc == 2) {
		options->last_img = options->skip_img;
		options->skip_img = 1;
	    } else if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-s")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%g" , &options->scale);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-z")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%g %g %g", 
			 &options->vol_size[0],
			 &options->vol_size[1],
			 &options->vol_size[2]);
	    if (rc == 1) {
		options->vol_size[1] = options->vol_size[0];
		options->vol_size[2] = options->vol_size[0];
	    } else if (rc != 3) {
		print_usage ();
	    }
	}
	else {
	    print_usage ();
	}
    }
}
