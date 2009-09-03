/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "drr_opts.h"
#include "mathutil.h"

#ifndef NULL
#define NULL ((void*)0)
#endif

void
print_usage (void)
{
    printf ("Usage: mghdrr [options] [infile]\n"
	    "Options:\n"
	    " -a num            Generate num equally spaced angles\n"
	    " -A ang            Difference between neighboring anges (in degrees)\n"
	    " -r \"r1 r2\"        Set output resolution (in pixels)\n"
	    " -s scale          Scale the intensity of the output file\n"
	    " -e                Do exponential mapping of output values\n"
	    " -g \"sad sid\"      Set the sad, sid (in mm)\n"
	    " -c \"c1 c2\"        Set the image center (in pixels)\n"
	    " -z \"s1 s2\"        Set the physical size of imager (in mm)\n"
	    " -w \"w1 w2 w3 w4\"  Only produce image for pixes in window (in pix)\n"
	    " -t outformat      Select output format: pgm, pfm or raw\n"
	    " -S                Output multispectral output files\n"
	    " -i exact          Use exact trilinear interpolation\n"
	    " -i approx         Use approximate trilinear interpolation\n"
	    " -o \"o1 o2 o3\"     Set isocenter position\n"
	    " -I infile         Set the input file in mha format\n"
	    " -O outprefix      Generate output files using the specified prefix\n"
	    );
    exit (1);
}

void
set_default_options (MGHDRR_Options* options)
{
    options->image_resolution[0] = 128;
    options->image_resolution[1] = 128;
    options->image_size[0] = 600;
    options->image_size[1] = 600;
    options->isocenter[0] = 0.0f;
    options->isocenter[1] = 0.0f;
    options->isocenter[2] = 0.0f;
    options->have_image_center = 0;
    options->have_image_window = 0;
    options->have_angle_diff = 0;
    options->num_angles = 1;
    options->angle_diff = 1.0f;
    options->sad = 1000.0f;
    options->sid = 1630.0f;
    options->scale = 1.0f;
    options->input_file = 0;
    options->output_prefix = "out_";
    options->exponential_mapping = 0;
    options->output_format= OUTPUT_FORMAT_PFM;
    options->multispectral = 0;
    options->interpolation = INTERPOLATION_NONE;
}

void
set_image_parms (MGHDRR_Options* options)
{
    if (!options->have_image_center) {
	options->image_center[0] = (options->image_resolution[0]-1)/2.0;
	options->image_center[1] = (options->image_resolution[1]-1)/2.0;
    }
    if (!options->have_image_window) {
	options->image_window[0] = 0;
	options->image_window[1] = options->image_resolution[0] - 1;
	options->image_window[2] = 0;
	options->image_window[3] = options->image_resolution[1] - 1;
    }
    if (options->have_angle_diff) {
	options->angle_diff *= (float) (M_TWOPI / 360.0);
    } else {
	options->angle_diff = M_TWOPI / options->num_angles;
    }
}

void
parse_args (MGHDRR_Options* options, int argc, char* argv[])
{
    int i, rc;

    set_default_options (options);
    for (i = 1; i < argc; i++) {
	//printf ("ARG[%d] = %s\n", i, argv[i]);
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-r")) {
	    i++;
	    rc = sscanf (argv[i], "%d %d", &options->image_resolution[0], 
			 &options->image_resolution[1]);
	    if (rc == 1) {
		options->image_resolution[1] = options->image_resolution[0];
	    } else if (rc != 2) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-I")) {
	    i++;
	    options->input_file = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-O")) {
	    i++;
	    options->output_prefix = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-a")) {
	    i++;
	    rc = sscanf (argv[i], "%d" , &options->num_angles);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-A")) {
	    i++;
	    rc = sscanf (argv[i], "%g" , &options->angle_diff);
	    if (rc != 1) {
		print_usage ();
	    }
	    options->have_angle_diff = 1;
	}
	else if (!strcmp (argv[i], "-s")) {
	    i++;
	    rc = sscanf (argv[i], "%g" , &options->scale);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-t")) {
	    i++;
	    if (!strcmp (argv[i], "pfm")) {
		options->output_format = OUTPUT_FORMAT_PFM;
	    }
	    else if (!strcmp (argv[i], "pgm")) {
		options->output_format = OUTPUT_FORMAT_PGM;
	    }
	    else if (!strcmp (argv[i], "raw")) {
		options->output_format = OUTPUT_FORMAT_RAW;
	    }
	    else {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-c")) {
	    i++;
	    rc = sscanf (argv[i], "%g %g", &options->image_center[0],
			 &options->image_center[1]);
	    if (rc == 1) {
		options->image_center[1] = options->image_center[0];
	    } else if (rc != 2) {
		print_usage ();
	    }
	    options->have_image_center = 1;
	}
	else if (!strcmp (argv[i], "-z")) {
	    i++;
	    rc = sscanf (argv[i], "%g %g", &options->image_size[0],
			 &options->image_size[1]);
	    if (rc == 1) {
		options->image_size[1] = options->image_size[0];
	    } else if (rc != 2) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-g")) {
	    i++;
	    rc = sscanf (argv[i], "%g %g", &options->sad, &options->sid);
	    if (rc != 2) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-w")) {
	    i++;
	    rc = sscanf (argv[i], "%d %d %d %d",
			 &options->image_window[0],
			 &options->image_window[1],
			 &options->image_window[2],
			 &options->image_window[3]);
	    if (rc == 2) {
		options->image_window[2] = options->image_window[0];
		options->image_window[3] = options->image_window[1];
	    } else if (rc != 4) {
		print_usage ();
	    }
	    options->have_image_window = 1;
	}
	else if (!strcmp (argv[i], "-i")) {
	    i++;
	    if (!strcmp(argv[i], "exact")) {
		options->interpolation = INTERPOLATION_TRILINEAR_EXACT;
	    } else if (!strcmp(argv[i], "approx")) {
		options->interpolation = INTERPOLATION_TRILINEAR_APPROX;
	    } else {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-o")) {
	    i++;
	    rc = sscanf (argv[i], "%g %g %g" , 
			 &options->isocenter[0],
			 &options->isocenter[1],
			 &options->isocenter[2]);
	    if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-S")) {
	    options->multispectral = 1;
	}
	else if (!strcmp (argv[i], "-e")) {
	    options->exponential_mapping = 1;
	}
	else {
	    print_usage ();
	    break;
	}
    }

    if (!options->input_file) {
	if (i < argc) {
	    options->input_file = strdup (argv[i++]);
	}
    }
    if (i < argc) {
	print_usage ();
    }
    if (!options->input_file) {
	print_usage ();
    }

    set_image_parms (options);
}
