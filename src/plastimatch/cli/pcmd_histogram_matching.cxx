/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "itk_histogram_matching.h"
#include "pcmd_histogram_matching.h"
#include "itk_image_type.h"
#include "plm_clp.h"
#include "plm_file_format.h"
#include "plm_image.h"
#include "print_and_exit.h"

class HistogramMatching_parms {
public:
    std::string output_fn;
    std::string reference_fn;
    std::string source_fn;

    bool threshold;
    int histogram_levels;
    int match_points;

public:
    HistogramMatching_parms () {
        threshold = false;
        histogram_levels = 128;
        match_points = 10;
    }
};

void
histogram_matching_main (HistogramMatching_parms *parms)
{
    if (parms->histogram_levels <= 0 || parms->match_points <= 0)
    {
        print_and_exit (
                "Error, you specified invalid number of levels (%d) or match_points(%d)",
                parms->histogram_levels, parms->match_points);
    }

    Plm_image::Pointer src = plm_image_load (parms->source_fn, PLM_IMG_TYPE_ITK_FLOAT);
    Plm_image::Pointer ref = plm_image_load (parms->reference_fn, PLM_IMG_TYPE_ITK_FLOAT);

    src->m_itk_float = itk_histogram_matching(src->m_itk_float, ref->m_itk_float, parms->threshold,
                parms->histogram_levels, parms->match_points);

    src->convert_to_original_type();
    src->save_image(parms->output_fn);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] source_file reference_file\n",
        argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    HistogramMatching_parms* parms,
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output file */
    parser->add_long_option ("", "output", "output image", 1, "");

    /* histogram levels */
    parser->add_long_option ("", "levels",
        "number of histogram bins for creating histograms (default 128)",
        1, "");

    /* match points */
    parser->add_long_option ("", "match-points",
        "number of quantile values to be matched (default 10)",
        1, "");

    parser->add_long_option("", "threshold",
        "threshold at mean intensity (simple background exclusion)",
        0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Check that no extraneous options were given */
    if (parser->number_of_arguments() != 2) {
	throw (dlib::error ("Error.  You must specify source and reference image"));
    }

    parms->source_fn = (*parser)[0];
    parms->reference_fn = (*parser)[1];

    /* threshold */
    if (parser->option ("threshold")) {
        parms->threshold = true;
    }

    /* Output file */
    parms->output_fn = parser->get_string("output");

    if (parser->option ("levels")) {
        parms->histogram_levels = parser->get_int("levels");
    }
    if (parser->option ("match-points")) {
        parms->match_points = parser->get_int("match-points");
    }
}

void
do_command_histogram_matching (int argc, char *argv[])
{
    HistogramMatching_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    histogram_matching_main (&parms);
}
