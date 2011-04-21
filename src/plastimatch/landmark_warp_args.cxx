/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "landmark_warp_args.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

void
print_usage (void)
{
    printf (
	"Usage: landmark_warp [options] moving\n"
	"Options:\n"
	"  -f infile                  Input fixed landmarks\n"
	"  -m infile                  Input moving landmarks\n"
	"  -v infile                  Input vector field\n"
	"  -x infile                  Input landmark xform\n"
	"  -O outfile                 Output warped image\n"
	"  -V outfile                 Output vector field\n"
	"  -a algorithm               Either \"itk\", \"gcs\", or \"nsh\"\n"
	"  -r float                   Radius of RBFs\n"
	"  -Y float                   Young modulus for RBF regularization\n"
	);
    exit (1);
}

void
Landmark_warp_args::parse_args (
    int argc, 
    char* argv[])
{
    int i;

    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
        if (!strcmp (argv[i], "-a")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "itk") || !strcmp(argv[i], "ITK")) {
		this->m_algorithm = LANDMARK_WARP_ALGORITHM_ITK_TPS;
	    }
	    else if (!strcmp(argv[i], "gcs") || !strcmp(argv[i], "GCS")) {
		this->m_algorithm = LANDMARK_WARP_ALGORITHM_RBF_GCS;
	    }
	    else if (!strcmp(argv[i], "nsh") || !strcmp(argv[i], "NSH")) {
		this->m_algorithm = LANDMARK_WARP_ALGORITHM_RBF_NSH;
	    }
	    else {
		printf ("Warning, unknown algorithm.  Defauling to GCS\n");
		this->m_algorithm = LANDMARK_WARP_ALGORITHM_RBF_GCS;
	    }
	}
        else if (!strcmp (argv[i], "-f")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    this->input_fixed_landmarks_fn = argv[i];
	}
        else if (!strcmp (argv[i], "-m")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    this->input_moving_landmarks_fn = argv[i];
	}
        else if (!strcmp (argv[i], "-x")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    this->input_xform_fn = argv[i];
	}
        else if (!strcmp (argv[i], "-v")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    this->input_vf_fn = argv[i];
	}
        else if (!strcmp (argv[i], "-O")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    this->output_warped_image_fn = argv[i];
	}
        else if (!strcmp (argv[i], "-V")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    this->output_vf_fn = argv[i];
	}
	else {
	    print_usage ();
	    break;
	}
    }
    if (i >= argc) {
	print_usage ();
    }
    this->input_moving_image_fn = argv[i];
    printf ("Moving = %s\n", (const char*) this->input_moving_image_fn);
}
