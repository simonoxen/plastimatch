/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <time.h>
#include "getopt.h"

#include "dvh.h"
#include "pcmd_dvh.h"
#include "plm_clp.h"
#include "pstring.h"

class Dvh_parms_pcmd {
public:
    Pstring input_ss_img_fn;
    Pstring input_ss_list_fn;
    Pstring input_dose_fn;
    Pstring output_csv_fn;
    Dvh::Dvh_units dose_units;
    Dvh::Dvh_normalization normalization;
    Dvh::Histogram_type histogram_type;
    int num_bins;
    float bin_width;
public:
    Dvh_parms_pcmd () {
        dose_units = Dvh::default_dose_units ();
        normalization = Dvh::default_normalization ();
        histogram_type = Dvh::default_histogram_type ();
        num_bins = Dvh::default_histogram_num_bins ();
        bin_width = Dvh::default_histogram_bin_width ();
    }
};

#if defined (commentout)
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
parse_args (Dvh_parms_pcmd* parms, int argc, char* argv[])
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
		parms->dose_units = Dvh::DVH_UNITS_CGY;
	    }
	    else if (!strcmp (optarg, "gy") || !strcmp (optarg, "Gy"))
	    {
		parms->dose_units = Dvh::DVH_UNITS_CGY;
	    }
	    else {
		fprintf (stderr, "Error.  Units must be Gy or cGy.\n");
		print_usage ();
	    }
	    break;
	case 7:
            parms->histogram_type = Dvh::DVH_CUMULATIVE_HISTOGRAM;
	    break;
	case 8:
	    rc = sscanf (optarg, "%d", &parms->num_bins);
	    std::cout << "num_bins " << parms->num_bins << "\n";
	    break;
	case 9:
	    rc = sscanf (optarg, "%f", &parms->bin_width);
	    std::cout << "bin_width " << parms->bin_width << "\n";
	    break;
	case 10:
	    if (!strcmp (optarg, "percent") || !strcmp (optarg, "pct"))
	    {
		parms->normalization = Dvh::DVH_NORMALIZATION_PCT;
	    }
	    else if (!strcmp (optarg, "voxels") || !strcmp (optarg, "vox"))
	    {
		parms->normalization = Dvh::DVH_NORMALIZATION_VOX;
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
    if (parms->input_ss_img_fn.empty()
	|| parms->input_dose_fn.empty()
        || parms->output_csv_fn.empty())
    {
	fprintf (stderr, 
	    "Error.  Must specify input for dose, ss_img, and output file.\n");
	print_usage ();
    }
}
#endif

static void
usage_fn (dlib::Plm_clp *parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch dvh [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Dvh_parms_pcmd *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char *argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files */
    parser->add_long_option ("", "input-ss-img", 
	"structure set image file", 1, "");
    parser->add_long_option ("", "input-ss-list", 
        "structure set list file containing names and colors", 1, "");
    parser->add_long_option ("", "input-dose", 
        "dose image file", 1, "");

    /* Parameters */
    parser->add_long_option ("", "input-units", 
        "specify units of dose in input file as either cGy as \"cgy\" "
        "or Gy as \"gy\" (default=\"gy\")", 1, "");
    parser->add_long_option ("", "cumulative", 
        "create a cumulative DVH (this is the default)", 0);
    parser->add_long_option ("", "differential", 
        "create a differential DVH instead of a cumulative DVH", 0);
    parser->add_long_option ("", "normalization", 
        "specify histogram values as either voxels \"vox\" or percent "
        "\"pct\" (default=\"pct\")", 1, "");
    parser->add_long_option ("", "num-bins", 
        "specify number of bins in the histogram (default=256)", 1, "");
    parser->add_long_option ("", "bin-width", 
        "specify bin width in the histogram in units of Gy "
        "(default=0.5)", 1, "");

    /* Output files */
    parser->add_long_option ("", "output-csv", 
        "file to save dose volume histogram data in csv format", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that input file were given */
    if (!parser->have_option ("input-dose"))
    {
	throw (dlib::error ("Error.  You must specify an input dose "
                " with --input-dose"));
    }
    if (!parser->have_option ("input-ss-img"))
    {
	throw (dlib::error ("Error.  You must specify an input structure "
                "set with --input-ss-img image"));
    }

    /* Check that an output file was given */
    if (!parser->have_option ("output-csv"))
    {
	throw (dlib::error (
                "Error.  You must specify an ouput file with --output-csv"));
    }

    /* Copy values into output struct */
    parms->input_ss_img_fn = parser->get_string("input-ss-img").c_str();
    if (parser->have_option ("input-ss-list")) {
        parms->input_ss_list_fn = parser->get_string("input-ss-list").c_str();
    }
    parms->input_dose_fn = parser->get_string("input-dose").c_str();
    parms->output_csv_fn = parser->get_string("output-csv").c_str();
    if (parser->have_option ("dose-units")) {
        if (parser->get_string("dose-units") == "cGy" 
            || parser->get_string("dose-units") == "cgy")
        {
            parms->dose_units = Dvh::DVH_UNITS_CGY;
        }
    }
    if (parser->have_option ("normalization")) {
        if (parser->get_string("normalization") == "vox") {
            parms->normalization = Dvh::DVH_NORMALIZATION_VOX;
        }
    }
    if (parser->have_option ("differential")) {
        parms->histogram_type = Dvh::DVH_DIFFERENTIAL_HISTOGRAM;
    }
    if (parser->have_option ("num-bins")) {
        parms->num_bins = parser->get_int ("num-bins");
    }
    if (parser->have_option ("bin-width")) {
        parms->bin_width = parser->get_float ("bin-width");
    }
}

void
do_command_dvh (int argc, char *argv[])
{
    Dvh_parms_pcmd parms;
    
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    Dvh dvh;
    dvh.set_structure_set_image (
        (const char*) parms.input_ss_img_fn, 
        (const char*) parms.input_ss_list_fn);
    dvh.set_dose_image (
        (const char*) parms.input_dose_fn);
    dvh.set_dose_units (parms.dose_units);
    dvh.set_dvh_parameters (
        parms.normalization,
        parms.histogram_type, 
        parms.num_bins,
        parms.bin_width);

    dvh.run ();

    dvh.save_csv ((const char*) parms.output_csv_fn);
}
