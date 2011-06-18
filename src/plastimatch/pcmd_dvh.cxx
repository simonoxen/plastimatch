/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include "getopt.h"

#include "bstring_util.h"
#include "dvh.h"
#include "pcmd_dvh.h"

static void
print_usage (void)
{
    printf (
	"Usage: plastimatch dvh [options]\n"
	"   --input-ss-img file\n"
	"   --input-ss-list file\n"
	"   --input-dose file\n"
	"   --output-csv file\n"
	"   --input-units {gy,cgy}\n"
	"   --cumulative\n"
	"   --normalization {pct,vox}\n"
	"   --num-bins\n"
	"   --bin-width (in cGy)\n"
    );
    exit (-1);
}

static void
parse_args (Dvh_parms* parms, int argc, char* argv[])
{
    int rc;
    int ch;
    static struct option longopts[] = {
	{ "input_ss_img",   required_argument,      NULL,           2 },
	{ "input-ss-img",   required_argument,      NULL,           2 },
	{ "input_ss_list",  required_argument,      NULL,           3 },
	{ "input-ss-list",  required_argument,      NULL,           3 },
	{ "input_dose",     required_argument,      NULL,           4 },
	{ "input-dose",     required_argument,      NULL,           4 },
	{ "output_csv",     required_argument,      NULL,           5 },
	{ "output-csv",     required_argument,      NULL,           5 },
	{ "input_units",    required_argument,      NULL,           6 },
	{ "input-units",    required_argument,      NULL,           6 },
	{ "cumulative",     no_argument,            NULL,           7 },
	{ "num_bins",       required_argument,      NULL,           8 },
	{ "num-bins",       required_argument,      NULL,           8 },
	{ "bin_width",      required_argument,      NULL,           9 },
	{ "bin-width",      required_argument,      NULL,           9 },
	{ "normalization",  required_argument,      NULL,           10 },
	{ NULL,             0,                      NULL,           0 }
    };

    /* Skip command "dvh" */
    optind ++;

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    parms->input_ss_img_fn = optarg;
	    break;
	case 3:
	    parms->input_ss_list_fn = optarg;
	    break;
	case 4:
	    parms->input_dose_fn = optarg;
	    break;
	case 5:
	    parms->output_csv_fn = optarg;
	    break;
	case 6:
	    if (!strcmp (optarg, "cgy") || !strcmp (optarg, "cGy"))
	    {
		parms->input_units = DVH_UNITS_CGY;
	    }
	    else if (!strcmp (optarg, "gy") || !strcmp (optarg, "Gy"))
	    {
		parms->input_units = DVH_UNITS_CGY;
	    }
	    else {
		fprintf (stderr, "Error.  Units must be Gy or cGy.\n");
		print_usage ();
	    }
	    break;
	case 7:
	    parms->cumulative = 1;
	    break;
	case 8:
	    rc = sscanf (optarg, "%d", &parms->num_bins);
	    std::cout << "num_bins " << parms->num_bins << std::endl;
	    break;
	case 9:
	    rc = sscanf (optarg, "%f", &parms->bin_width);
	    std::cout << "bin_width " << parms->bin_width << std::endl;
	    break;
	case 10:
	    if (!strcmp (optarg, "percent") || !strcmp (optarg, "pct"))
	    {
		parms->normalization = DVH_NORMALIZATION_PCT;
	    }
	    else if (!strcmp (optarg, "voxels") || !strcmp (optarg, "vox"))
	    {
		parms->normalization = DVH_NORMALIZATION_VOX;
	    }
	    else {
		fprintf (stderr, "Error.  Normalization must be pct or vox.\n");
		print_usage ();
	    }
	    break;
	default:
	    fprintf (stderr, "Error.  Unknown option.\n");
	    print_usage ();
	    break;
	}
    }
    if (bstring_empty (parms->input_ss_img_fn)
	|| bstring_empty (parms->input_ss_list_fn)
	|| bstring_empty (parms->input_dose_fn)
	|| bstring_empty (parms->output_csv_fn))
    {
	fprintf (stderr, 
	    "Error.  Must specify input for dose, ss_img, and output file.\n");
	print_usage ();
    }
}

void
do_command_dvh (int argc, char *argv[])
{
    Dvh_parms parms;
    
    parse_args (&parms, argc, argv);

    dvh_execute (&parms);
}
