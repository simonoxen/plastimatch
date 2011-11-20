/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "fdk_opts.h"
#include "volume.h"

void 
print_usage (void)
{
    printf (
	"Usage: fdk [options]\n"
	"Options:\n"
	" -A hardware            One of \"cpu\", \"cuda\" or \"opencl\" (default=cpu)\n"
	" -a \"num ((num) num)\"   Use this range of images\n"
	" -r \"r1 r2 r3\"          Set output resolution (in voxels)\n"
	" -f filter              Either \"none\" or \"ramp\" (default=ramp)\n"
	" -s scale               Scale the intensity of the output file\n"
	" -z \"s1 s2 s3\"          Physical size of the reconstruction (in mm)\n"
	" -I indir               The input directory\n"
	" -O outfile             The output file\n"
        " -x \"x0 y0\"           Panel offset (in pixels)\n"
        " -X flavor              Implementation flavor (0,a,b,c,d) (default=c)\n"
    );
    exit (1);
}

void 
set_default_options (Fdk_options* options)
{
    options->threading = THREADING_CPU_OPENMP;
    options->image_range_requested = 0;
    options->first_img = 0;
    options->last_img = 119;
    options->skip_img = 1;
    options->resolution[0] = 256;
    options->resolution[1] = 256;
    options->resolution[2] = 100;
    options->vol_size[0] = 300.0f;
    options->vol_size[1] = 300.0f;
    options->vol_size[2] = 150.0f;
    options->xy_offset[0] = 0.f;
    options->xy_offset[1] = 0.f;
    options->scale = 1.0f;
    options->filter = FDK_FILTER_TYPE_RAMP;
    options->input_dir = ".";
    options->output_file = "output.mha";
    options->flavor = 'c';
    options->full_fan=1;
    options->Full_normCBCT_name="Full_norm.mh5";
    options->Full_radius=120;
    options->Half_normCBCT_name="Half_norm.mh5";
    options->Half_radius=220;
}

void 
fdk_parse_args (Fdk_options* options, int argc, char* argv[])
{
    int i, rc;
	
    if (argc < 2)
    { print_usage(); exit(1); }

    set_default_options (options);
    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-A")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
#if CUDA_FOUND
	    if (!strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")) {
		options->threading = THREADING_CUDA;
		continue;
	    }
#endif
#if OPENCL_FOUND
	    if (!strcmp(argv[i], "opencl") || !strcmp(argv[i], "OPENCL")) {
		options->threading = THREADING_OPENCL;
		continue;
	    }
#endif
	    /* Default */
	    options->threading = THREADING_CPU_OPENMP;
	}
	else if (!strcmp (argv[i], "-a")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->image_range_requested = 1;
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
	else if (!strcmp (argv[i], "-f")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "none") || !strcmp(argv[i], "NONE")) {
		options->filter = FDK_FILTER_TYPE_NONE;
	    }
	    else if (!strcmp(argv[i], "ramp") || !strcmp(argv[i], "RAMP")) {
		options->filter = FDK_FILTER_TYPE_RAMP;
	    }
	    else {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-X")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->flavor = argv[i][0];
	    if (options->flavor != '0' && options->flavor != 'a'
		&& options->flavor != 'b' && options->flavor != 'c'
		&& options->flavor != 'd') {
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
	else if (!strcmp (argv[i], "-r")) {
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
	else if (!strcmp (argv[i], "-x")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%f %f", 
		&options->xy_offset[0],
		&options->xy_offset[1]);
	    if (rc != 2) {
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
