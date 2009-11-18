/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <time.h>
#include "plm_config.h"

#include "getopt.h"
#include "warp_main.h"
#include "file_type.h"
#include "warp_main.h"
#include "print_and_exit.h"

static void
warp_print_usage ()
{
    printf ("Usage: plastimatch warp [options]\n"
	    "Required:\n"
	    "    --input=filename\n"
	    "    --output=filename\n"
	    "Optional:\n"
	    "    --xf=filename\n"
	    "    --interpolation=nn\n"
	    "    --fixed=filename\n"
	    "    --offset=\"x y z\"\n"
	    "    --spacing=\"x y z\"\n"
	    "    --dims=\"x y z\"\n"
	    "    --output-vf=filename\n"
	    "    --default-val=number\n"
	    "    --output-format=dicom\n"
	    "    --algorithm=itk\n"
	    "    --ctatts=filename  (for dij)\n"
	    "    --dif=filename     (for dij)\n"
	    );
    exit (-1);
}

static void
convert_print_usage ()
{
    printf ("Usage: plastimatch convert [options]\n"
	    "Required:\n"
	    "    --input=filename\n"
	    "    --output=filename\n"
	    "Optional:\n"
	    "    --xf=filename\n"
	    "    --interpolation=nn\n"
	    "    --fixed=filename\n"
	    "    --offset=\"x y z\"\n"
	    "    --spacing=\"x y z\"\n"
	    "    --dims=\"x y z\"\n"
	    "    --output-vf=filename\n"
	    "    --default-val=number\n"
	    "    --output-format=dicom\n"
	    "    --algorithm=itk\n");
    exit (-1);
}

static void
print_usage (char* command)
{
    if (!strcmp (command, "convert")) {
	convert_print_usage ();
    } else {
	warp_print_usage ();
    }
}

void
warp_parse_args (Warp_Parms* parms, int argc, char* argv[])
{
    int ch;
    int rc;
    int have_offset = 0;
    int have_spacing = 0;
    int have_dims = 0;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "vf",             required_argument,      NULL,           4 },
	{ "default_val",    required_argument,      NULL,           5 },
	{ "default-val",    required_argument,      NULL,           5 },
	{ "xf",             required_argument,      NULL,           6 },
	{ "fixed",	    required_argument,      NULL,           7 },
	{ "output_vf",      required_argument,      NULL,           8 },
	{ "output-vf",      required_argument,      NULL,           8 },
	{ "interpolation",  required_argument,      NULL,           9 },
	{ "offset",         required_argument,      NULL,           10 },
	{ "spacing",        required_argument,      NULL,           11 },
	{ "dims",           required_argument,      NULL,           12 },
	{ "output_format",  required_argument,      NULL,           13 },
	{ "output-format",  required_argument,      NULL,           13 },
	{ "ctatts",         required_argument,      NULL,           14 },
	{ "dif",            required_argument,      NULL,           15 },
	{ "algorithm",      required_argument,      NULL,           16 },
	{ NULL,             0,                      NULL,           0 }
    };

    /* Skip command "warp" */
    optind ++;

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    strncpy (parms->mha_in_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->mha_out_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (parms->vf_in_fn, optarg, _MAX_PATH);
	    break;
	case 5:
	    if (sscanf (optarg, "%f", &parms->default_val) != 1) {
		printf ("Error: default_val takes an argument\n");
		print_usage (argv[1]);
	    }
	    break;
	case 6:
	    strncpy (parms->xf_in_fn, optarg, _MAX_PATH);
	    break;
	case 7:
	    strncpy (parms->fixed_im_fn, optarg, _MAX_PATH);
	    break;
	case 8:
	    strncpy (parms->vf_out_fn, optarg, _MAX_PATH);
	    break;
	case 9:
	    if (!strcmp (optarg, "nn")) {
		parms->interp_lin = 0;
	    } else if (!strcmp (optarg, "linear")) {
		parms->interp_lin = 1;
	    } else {
		fprintf (stderr, "Error.  --interpolation must be either nn or linear.\n");
		print_usage (argv[1]);
	    }
	    break;
	case 10:
	    rc = sscanf (optarg, "%f %f %f", &parms->offset[0], &parms->offset[1], &parms->offset[2]);
	    if (rc != 3) {
		fprintf (stderr, "Error.  --offset requires 3 values.");
		print_usage (argv[1]);
	    }
	    have_offset = 1;
	    break;
	case 11:
	    rc = sscanf (optarg, "%f %f %f", &parms->spacing[0], &parms->spacing[1], &parms->spacing[2]);
	    if (rc != 3) {
		fprintf (stderr, "Error.  --spacing requires 3 values.");
		print_usage (argv[1]);
	    }
	    have_spacing = 1;
	    break;
	case 12:
	    rc = sscanf (optarg, "%d %d %d", &parms->dims[0], &parms->dims[1], &parms->dims[2]);
	    if (rc != 3) {
		fprintf (stderr, "Error.  --dims requires 3 values.");
		print_usage (argv[1]);
	    }
	    have_dims = 1;
	    break;
	case 13:
	    if (!strcmp (optarg, "dicom")) {
		parms->output_dicom = 1;
	    } else {
		fprintf (stderr, "Error.  --output-format option only supports dicom.\n");
		print_usage (argv[1]);
	    }
	    break;
	case 14:
	    strncpy (parms->ctatts_in_fn, optarg, _MAX_PATH);
	    break;
	case 15:
	    strncpy (parms->dif_in_fn, optarg, _MAX_PATH);
	    break;
	case 16:
	    if (!strcmp (optarg, "itk")) {
		parms->use_itk = 1;
	    } else {
		fprintf (stderr, "Error.  --algorithm option only supports itk.\n");
		print_usage (argv[1]);
	    }
	    break;
	default:
	    fprintf (stderr, "Error.  Unknown option.");
	    print_usage (argv[1]);
	    break;
	}
    }
    if (!parms->mha_in_fn[0] || !parms->mha_out_fn[0]) {
	print_usage (argv[1]);
    }
}

void
do_command_warp (int argc, char* argv[])
{
    Warp_Parms parms;
    File_type file_type;

    void test_fn (Warp_Parms *parms);
    
    warp_parse_args (&parms, argc, argv);
    file_type = deduce_file_type (parms.mha_in_fn);

    if (parms.ctatts_in_fn[0] && parms.dif_in_fn[0]) {
	warp_dij_main (&parms);
    } else {
	switch (file_type) {
	case FILE_TYPE_NO_FILE:
	    print_and_exit ("Could not open input file %s for read\n",
		parms.mha_in_fn);
	    break;
	case FILE_TYPE_UNKNOWN:
	case FILE_TYPE_IMG:
	case FILE_TYPE_DICOM_DIR:
	    warp_image_main (&parms);
	    break;
	case FILE_TYPE_DIJ:
	    print_and_exit ("Warping dij files requres ctatts_in and dif_in files\n");
	    break;
	case FILE_TYPE_POINTSET:
	    warp_pointset_main (&parms);
	    break;
	default:
	    print_and_exit ("Sorry, don't know how to warp input type %s (%s)\n",
			    file_type_string (file_type),
			    parms.mha_in_fn);
	    break;
	}
    }

    printf ("Finished!\n");
}
