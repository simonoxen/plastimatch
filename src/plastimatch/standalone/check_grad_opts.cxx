/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "check_grad_opts.h"

static void
print_usage (void)
{
    printf (
	"Usage: check_grad [options] fixed moving\n"
	"Options:\n"
	" -A hardware             Either \"cpu\" or \"cuda\" (default=cpu)\n"
	" -M { mse | mi }         Registration metric (default is mse)\n"
	" -f implementation       Choose implementation (a single letter: a, b, etc.)\n"
	" -s \"i j k\"              Integer knot spacing (voxels)\n"
	" -h prefix               Generate histograms for each MI iteration\n"
	" --debug                 Create various debug files\n"
	" -p process              Choices: \"fwd\", \"bkd\", \"ctr\" (for forward,\n"
	"                           backward, or central difference, or \"line\" for\n"
	"                           line profile. (default=fwd)\n"
	" -e step                 Step size (default is 1e-4)\n"
	" -l \"min max\"            Min, max range for line profile (default \"0 30\")\n"
	" -R \"min max\"          Random starting point (coeff between min, max)\n"
	" -X infile               Input bspline coefficients\n"
	" -O file                 Output file\n"
    );
    exit (1);
}

void
check_grad_opts_parse_args (Check_grad_opts* options, 
    int argc, char* argv[])
{
    int i, rc;
    Bspline_parms* parms = &options->parms;

    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-A")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")
		|| !strcmp(argv[i], "gpu") || !strcmp(argv[i], "GPU")) {
		parms->threading = BTHR_CUDA;
	    }
	    else {
		parms->threading = BTHR_CPU;
	    }
	}
	else if (!strcmp (argv[i], "-f")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->implementation = argv[i][0];
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
	    unsigned int a, b, c;
	    rc = sscanf (argv[i], "%d %d %d", &a, &b, &c);
	    if (rc == 1) {
		options->vox_per_rgn[0] = a;
		options->vox_per_rgn[1] = a;
		options->vox_per_rgn[2] = a;
	    } else if (rc == 3) {
		options->vox_per_rgn[0] = a;
		options->vox_per_rgn[1] = b;
		options->vox_per_rgn[2] = c;
	    } else {
		print_usage ();
	    }
	}
        else if (!strcmp (argv[i], "-h")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->xpm_hist_dump = strdup (argv[i]);
	}
        else if (!strcmp (argv[i], "--debug")) {
	    parms->debug = 1;
	}
        else if (!strcmp (argv[i], "-e")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%g", &options->step_size);
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
	    if (!strcmp(argv[i], "fwd")) {
		options->process = CHECK_GRAD_PROCESS_FWD;
	    } else if (!strcmp(argv[i], "bkd")) {
		options->process = CHECK_GRAD_PROCESS_BKD;
	    } else if (!strcmp(argv[i], "ctr")) {
		options->process = CHECK_GRAD_PROCESS_CTR;
	    } else if (!strcmp(argv[i], "line")) {
		options->process = CHECK_GRAD_PROCESS_LINE;
	    } else {
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
        else if (!strcmp (argv[i], "-R")) {
	    i++;
	    rc = sscanf (argv[i], "%g %g", 
		&options->random_range[0],
		&options->random_range[1]);
	    if (rc != 2) {
		printf ("Error parsing -R option.\n");
		print_usage ();
	    }
	    options->random = 1;
	}
        else if (!strcmp (argv[i], "-l")) {
	    i++;
	    rc = sscanf (argv[i], "%d %d", 
		&options->line_range[0],
		&options->line_range[1]);
	    if (rc != 2) {
		printf ("Error parsing -l option.\n");
		print_usage ();
	    }
	}
        else if (!strcmp (argv[i], "-X")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->input_xf_fn = strdup (argv[i]);
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

    if (!options->output_fn) {
	options->output_fn = "check_grad.txt";
    }
}
