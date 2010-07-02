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
	" -A hardware            Either \"cpu\" or \"brook\" or \"cuda\" (default=cpu)\n"
	" -a \"num ((num) num)\"   Use this range of images\n"
	" -r \"r1 r2 r3\"          Set output resolution (in voxels)\n"
	" -f filter              Either \"none\" or \"ramp\" (default=ramp)\n"
	" -s scale               Scale the intensity of the output file\n"
	" -z \"s1 s2 s3\"          Physical size of the reconstruction (in mm)\n"
	" -I indir               The input directory\n"
	" -O outfile             The output file\n"
	" -sb \". (default)\" The subfolder with *.raw files\n"
	" -F \"F(f)ull (default)\"  or \"H(alf)\"     Full/Half fan options\n"
	" -cor                   Turn on Coronal output\n"
	" -sag                   Turn on Sagittal output\n"
    );
    exit (1);
}

void 
set_default_options (Fdk_options* options)
{
    options->threading = THREADING_CPU;
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
    options->scale = 1.0f;
    options->filter = FDK_FILTER_TYPE_RAMP;
    options->input_dir = ".";
    options->output_file = "output.mha";

    options->full_fan=1;
    options->coronal=0;
    options->sagittal=0;
    options->sub_dir = ".";
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
	    if (!strcmp(argv[i], "brook") || !strcmp(argv[i], "BROOK")) {
		options->threading = THREADING_BROOK;
	    } 
	    else if (!strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")
		     || !strcmp(argv[i], "gpu") || !strcmp(argv[i], "GPU")) {
		options->threading = THREADING_CUDA;
	    }
	    else {
		options->threading = THREADING_CPU;
	    }
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
	else if (!strcmp (argv[i], "-cor")) {
	    options->coronal=1;
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
	else if (!strcmp (argv[i], "-F")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i],"FULL")||!strcmp(argv[i],"full")|!strcmp(argv[i],"Full"))
		options->full_fan=1;
	    else if (!strcmp(argv[i],"HALF")||!strcmp(argv[i],"half")||!strcmp(argv[i],"Half"))
		options->full_fan=0;
	    else 
		print_usage ();
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
	else if (!strcmp (argv[i], "-sag")) {
	    options->sagittal=1;
	}
	else if (!strcmp (argv[i], "-sb")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->sub_dir = strdup (argv[i]);
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
