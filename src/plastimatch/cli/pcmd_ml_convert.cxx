/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "ml_convert.h"
#include "pcmd_ml_convert.h"
#include "plm_clp.h"

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch ml_convert [options] file\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Ml_convert* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "feature-directory",
	"Location of feature directory, one image per feature", 1, "");
    parser->add_long_option ("", "labelmap",
	"Location of labelmap file", 1, "");
    parser->add_long_option ("", "output",
	"Location of output file to be written", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an index or location was given */
    if (!parser->have_option ("feature-directory") 
	|| !parser->have_option("labelmap")
	|| !parser->have_option("output"))
    {
	throw (dlib::error ("Error.  Must specify --feature-directory, "
                "--labelmap, and --output options"));
    }

    /* Copy values into output struct */
    parms->set_feature_directory (parser->get_string ("feature-directory"));
    parms->set_label_filename (parser->get_string ("labelmap"));
    parms->set_output_filename (parser->get_string ("output"));
}

void
do_command_ml_convert (int argc, char *argv[])
{
    Ml_convert ml_convert;

    plm_clp_parse (&ml_convert, &parse_fn, &usage_fn, argc, argv, 1);

    ml_convert.run();
}
