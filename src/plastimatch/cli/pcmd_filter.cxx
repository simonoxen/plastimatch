/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "logfile.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "pcmd_filter.h"
#include "print_and_exit.h"

static void
filter_main (Filter_parms* parms)
{
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] input_image\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Filter_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "output", "output image filename", 1, "");

    /* Filter options */
    parser->add_long_option ("", "kernel", "kernel image filename", 1, "");
    parser->add_long_option ("", "gauss-width",
        "the width (in mm) of a uniform Gaussian smoothing filter", 1, "10.");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that two input files were given */
    if (parser->number_of_arguments() != 1) {
	throw (dlib::error ("Error.  You must specify one input file"));
    }

    /* Input files */
    parms->in_image_fn = (*parser)[0].c_str();

    /* Output files */
    if (parser->option ("output")) {
        parms->out_image_fn = parser->get_string("output");
    }

    /* Filter options */
    if (parser->option ("kernel")) {
        parms->out_image_fn = parser->get_string("kernel");
    }
    if (parser->option ("gauss-width")) {
        parms->gauss_width = parser->get_float("gauss-width");
    }
}

void
do_command_filter (int argc, char *argv[])
{
    Filter_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    filter_main (&parms);
}
