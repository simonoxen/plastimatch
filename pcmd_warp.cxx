/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <time.h>

#include "bstring_util.h"
#include "cxt_io.h"
#include "cxt_to_mha.h"
#include "file_util.h"
#include "gdcm_rtss.h"
#include "getopt.h"
#include "plm_file_format.h"
#include "plm_image_header.h"
#include "plm_image_patient_position.h"
#include "plm_warp.h"
#include "print_and_exit.h"
#include "rtds.h"
#include "rtds_warp.h"
#include "pcmd_warp.h"
#include "xform.h"
#include "xio_structures.h"

static void
print_usage (char* command)
{
    printf (
	"Usage: plastimatch %s [options]\n"
	"Options:\n"
	"    --input=filename\n"
	"    --xf=filename\n"
	"    --interpolation=nn\n"
	"    --fixed=filename\n"
	"    --offset=\"x y z\"\n"
	"    --spacing=\"x y z\"\n"
	"    --dims=\"x y z\"\n"
	"    --default-val=number\n"
	"    --output-type={uchar,short,float,...}\n"
	"    --algorithm=itk\n"
	"    --patient-pos={hfs,hfp,ffs,ffp}\n"
	//	"    --dicom-dir=directory      (for structure association)\n"
	"    --referenced-ct=directory  (for structure association)\n"
	"    --ctatts=filename          (for dij)\n"
	"    --dif=filename             (for dij)\n"
	"    --input-ss-img=filename    (for structures)\n"
	"    --input-ss-list=filename   (for structures)\n"
	"    --prune-empty              (for structures)\n"
	"    --input-dose-img=filename  (for rt dose)\n"
	"    --input-dose-xio=filename  (for XiO rt dose)\n"
	"    --input-dose-ast=filename  (for Astroid rt dose)\n"
	"    --input-dose-mc=filename   (for Monte Carlo 3ddose rt dose)\n"
	"\n"
	"    --output-cxt=filename      (for structures)\n"
	"    --output-dicom=directory   (for image and structures)\n"
	"    --output-dij=filename      (for dij)\n"
	"    --output-dose-img=filename (for rt dose)\n"
	"    --output-img=filename      (for image)\n"
	"    --output-labelmap=filename (for structures)\n"
	"    --output-colormap=filename (for structures)\n"
	"    --output-prefix=string     (for structures)\n"
	"    --output-ss-img=filename   (for structures)\n"
	"    --output-ss-list=filename  (for structures)\n"
	"    --output-vf=filename       (for vector field)\n"
	"    --output-xio=directory     (for rt dose and structures)\n"
	,
	command);
    exit (-1);
}

void
warp_parse_args (Warp_parms* parms, int argc, char* argv[])
{
    int ch;
    int rc;
    int have_offset = 0;
    int have_spacing = 0;
    int have_dims = 0;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "output_img",     required_argument,      NULL,           3 },
	{ "output-img",     required_argument,      NULL,           3 },
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
	{ "output_dicom",   required_argument,      NULL,           13 },
	{ "output-dicom",   required_argument,      NULL,           13 },
	{ "ctatts",         required_argument,      NULL,           14 },
	{ "dif",            required_argument,      NULL,           15 },
	{ "algorithm",      required_argument,      NULL,           16 },
	{ "output-type",    required_argument,      NULL,           17 },
	{ "output_type",    required_argument,      NULL,           17 },
	{ "dicom-dir",      required_argument,      NULL,           18 },
	{ "dicom_dir",      required_argument,      NULL,           18 },
	{ "referenced-ct",  required_argument,      NULL,           18 },
	{ "referenced_ct",  required_argument,      NULL,           18 },
	{ "output-prefix",  required_argument,      NULL,           19 },
	{ "output_prefix",  required_argument,      NULL,           19 },
	{ "output-labelmap",required_argument,      NULL,           20 },
	{ "output_labelmap",required_argument,      NULL,           20 },
	{ "output-ss-img",  required_argument,      NULL,           21 },
	{ "output_ss_img",  required_argument,      NULL,           21 },
	{ "output_ss_list", required_argument,      NULL,           22 },
	{ "output-ss-list", required_argument,      NULL,           22 },
	{ "output_cxt",     required_argument,      NULL,           23 },
	{ "output-cxt",     required_argument,      NULL,           23 },
	{ "prune_empty",    required_argument,      NULL,           24 },
	{ "prune-empty",    no_argument,            NULL,           24 },
	{ "output_xio",     required_argument,      NULL,           25 },
	{ "output-xio",     required_argument,      NULL,           25 },
	{ "input_ss_list",  required_argument,      NULL,           26 },
	{ "input-ss-list",  required_argument,      NULL,           26 },
	{ "output_dij",     required_argument,      NULL,           27 },
	{ "output-dij",     required_argument,      NULL,           27 },
	{ "input_ss_img",   required_argument,      NULL,           28 },
	{ "input-ss-img",   required_argument,      NULL,           28 },
	{ "input_dose_img", required_argument,      NULL,           29 },
	{ "input-dose-img", required_argument,      NULL,           29 },
	{ "input_dose_xio", required_argument,      NULL,           30 },
	{ "input-dose-xio", required_argument,      NULL,           30 },
	{ "output_dose_img", required_argument,     NULL,           31 },
	{ "output-dose-img", required_argument,     NULL,           31 },
	{ "input_dose_ast", required_argument,      NULL,           32 },
	{ "input-dose-ast", required_argument,      NULL,           32 },
	{ "patient_pos",    required_argument,      NULL,           33 },
	{ "patient-pos",    required_argument,      NULL,           33 },
	{ "input_dose_mc",  required_argument,      NULL,           34 },
	{ "input-dose-mc",  required_argument,      NULL,           34 },
	{ "output_colormap",required_argument,      NULL,           35 },
	{ "output-colormap",required_argument,      NULL,           35 },
	{ NULL,             0,                      NULL,           0 }
    };

    /* Skip command "warp" */
    optind ++;

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    parms->input_fn = optarg;
	    break;
	case 3:
	    parms->output_img_fn = optarg;
	    break;
	case 4:
	    parms->vf_in_fn = optarg;
	    break;
	case 5:
	    if (sscanf (optarg, "%f", &parms->default_val) != 1) {
		printf ("Error: default_val takes an argument\n");
		print_usage (argv[1]);
	    }
	    break;
	case 6:
	    parms->xf_in_fn = optarg;
	    break;
	case 7:
	    parms->fixed_im_fn = optarg;
	    break;
	case 8:
	    parms->output_vf_fn = optarg;
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
	    parms->output_dicom = optarg;
	    break;
	case 14:
	    parms->ctatts_in_fn = optarg;
	    break;
	case 15:
	    parms->dif_in_fn = optarg;
	    break;
	case 16:
	    if (!strcmp (optarg, "itk")) {
		parms->use_itk = 1;
	    } else {
		fprintf (stderr, "Error.  --algorithm option only supports itk.\n");
		print_usage (argv[1]);
	    }
	    break;
	case 17:
	    parms->output_type = plm_image_type_parse (optarg);
	    if (parms->output_type == PLM_IMG_TYPE_UNDEFINED) {
		fprintf (stderr, "Error, unknown output type %s\n", optarg);
		print_usage (argv[1]);
	    }
	    break;
	case 18:
	    parms->referenced_dicom_dir = optarg;
	    break;
	case 19:
	    parms->output_prefix = optarg;
	    break;
	case 20:
	    parms->output_labelmap_fn = optarg;
	    break;
	case 21:
	    parms->output_ss_img_fn = optarg;
	    break;
	case 22:
	    parms->output_ss_list_fn = optarg;
	    break;
	case 23:
	    parms->output_cxt_fn = optarg;
	    break;
	case 24:
	    parms->prune_empty = 1;
	    break;
	case 25:
	    parms->output_xio_dirname = optarg;
	    break;
	case 26:
	    parms->input_ss_list_fn = optarg;
	    break;
	case 27:
	    parms->output_dij_fn = optarg;
	    break;
	case 28:
	    parms->input_ss_img_fn = optarg;
	    break;
	case 29:
	    parms->input_dose_img_fn = optarg;
	    break;
	case 30:
	    parms->input_dose_xio_fn = optarg;
	    break;
	case 31:
	    parms->output_dose_img_fn = optarg;
	    break;
	case 32:
	    parms->input_dose_ast_fn = optarg;
	    break;
	case 33:
	    parms->patient_pos = plm_image_patient_position_parse (optarg);
	    if (parms->patient_pos == PATIENT_POSITION_UNKNOWN) {
		fprintf (stderr, "Error.  Unknown patient position {hfs,hfp,ffs,ffp}.\n");
		print_usage (argv[1]);
	    }
	    break;
	case 34:
	    parms->input_dose_mc_fn = optarg;
	    break;
	case 35:
	    parms->output_colormap_fn = optarg;
	    break;
	default:
	    fprintf (stderr, "Error.  Unknown option.\n");
	    print_usage (argv[1]);
	    break;
	}
    }
    if (bstring_empty (parms->input_fn)
	&& bstring_empty (parms->input_ss_img_fn)
	&& bstring_empty (parms->input_dose_img_fn)
	&& bstring_empty (parms->input_dose_xio_fn)
	&& bstring_empty (parms->input_dose_ast_fn)
	&& bstring_empty (parms->input_dose_mc_fn))
    {
	fprintf (stderr, "Error.  No input file specified..\n");
	print_usage (argv[1]);
    }
}

void
do_command_warp (int argc, char* argv[])
{
    Warp_parms parms;
    Plm_file_format file_type;
    Rtds rtds;

    /* Parse command line parameters */
    warp_parse_args (&parms, argc, argv);

    /* Dij matrices are a special case */
    if (bstring_not_empty (parms.output_dij_fn)) {
	if (bstring_not_empty (parms.ctatts_in_fn)
	    && bstring_not_empty (parms.dif_in_fn))
	{
	    warp_dij_main (&parms);
	    return;
	} else {
	    print_and_exit ("Sorry, you need to specify --ctatts and --dif for dij warping.\n");
	}
    }

    /* What is the input file type? */
    file_type = plm_file_format_deduce ((const char*) parms.input_fn);

    /* Pointsets are a special case */
    if (file_type == PLM_FILE_FMT_POINTSET) {
	warp_pointset_main (&parms);
	return;
    }

    /* Process warp */
    rtds_warp (&rtds, file_type, &parms);

    printf ("Finished!\n");
}
