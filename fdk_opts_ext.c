/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "fdk_opts_ext.h"
#include "volume.h"


void print_usage (void)
{
    printf ("Usage: mghcbct [options]\n"
	    "Options:\n"
	    " -a \"num ((num) num)\"   Use this range of images\n"
	    " -r \"r1 r2 r3\"          Set output resolution (in voxels)\n"
	    " -s scale               Scale the intensity of the output file\n"
	    " -z \"s1 s2 s3\"          Physical size of the reconstruction (in mm)\n"
		" -I indir               The input (parent) directory containing ProjAngles.txt\n"
		" -sb \". (default)\" The subfolder with *.raw files\n"
		" -O outfile             The output file. \n"
		"    (In DRR mode, this is input mha images with 512-byte file header. )\n"
		" -F \"F(f)ull (default)\"  or \"H(alf)\"     Full/Half fan options\n"
		" -cor (=0 default)                      Turn on Coronal output   \n"
		" -sag (=0 default)                     Turn on Sagittal output \n"
		" -DRR					        Generate DRR instead of FDK\n"     
		" In DRR mode, DRR files will be generated in side the indir\\DRR\n"
		" Subdir DRR will be automatically in windows but need to be create in linux"
	    );
    exit (1);
}

void set_default_options_ext (MGHCBCT_Options_ext* options)
{
	options->first_img = 0;
    options->last_img = 119;
    options->resolution[0] = 256;
    options->resolution[1] = 256;
    options->resolution[2] = 100;
    options->vol_size[0] = 300.0f;
    options->vol_size[1] = 300.0f;
    options->vol_size[2] = 150.0f;
    options->scale = 1.0f;
	options->full_fan=1;
	options->coronal=0;
	options->sagittal=0;
	options->DRR=0;
    //options->input_dir = ".";
	options->sub_dir = ".";
    options->output_file = "output.mh5";
	options->Full_normCBCT_name="Full_norm.mh5"; 
	options->Full_radius=120;
	options->Half_normCBCT_name="Half_norm.mh5"; 
	options->Half_radius=220;
    //options->first_img = 0;
    //options->last_img = 119;
    //options->resolution[0] = 120;
    //options->resolution[1] = 120;
    //options->resolution[2] = 120;
    //options->vol_size[0] = 500.0f;
    //options->vol_size[1] = 500.0f;
    //options->vol_size[2] = 500.0f;
    //options->scale = 1.0f;
    //options->input_dir = ".";
    //options->output_file = "output.mha";
}

void parse_args_ext (MGHCBCT_Options_ext* options, int argc, char* argv[])
{
    int i, rc;
	
    if (argc < 2)
	{ print_usage(); exit(1); }

    set_default_options_ext (options);
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
	else if (!strcmp (argv[i], "-cor")) {
		options->coronal=1;
	}
	else if (!strcmp (argv[i], "-sag")) {
		options->sagittal=1;
	}
	else if (!strcmp (argv[i], "-DRR")) {
		options->DRR=1;
	}
	else if (!strcmp (argv[i], "-I")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->input_dir = strdup (argv[i]);
	}
		else if (!strcmp (argv[i], "-sb")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    options->sub_dir = strdup (argv[i]);
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
	if (!options->input_dir){
		print_usage ();
	}

}
