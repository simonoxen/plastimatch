/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include "itkImageRegionIterator.h"

#include "autolabel.h"
#include "bstring_util.h"
#include "plm_clp.h"
#include "print_and_exit.h"
#include "pstring.h"

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch autolabel [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Autolabel_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "output-csv", 
	"Output csv filename", 1, "");
    parser->add_long_option ("", "output-fcsv", 
	"Output fcsv filename", 1, "");
    parser->add_long_option ("", "input", 
	"Input image filename (required)", 1, "");
    parser->add_long_option ("", "network", 
	"Input trained network filename (required)", 1, "");
    parser->add_long_option ("", "eac", 
	"Enforce anatomic constraints", 0);
    parser->add_long_option ("", "task", 
	"Labeling task (required), choices are "
	"{la,tsv1,tsv2}", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an input file was given */
    parser->check_required ("input");

    /* Check that an output file was given */
    if (!parser->option("output-csv") && !parser->option("output-fcsv")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using one of the --output options"));
    }

    /* Check that an network file was given */
    parser->check_required ("network");

    /* Check that a task was given */
    parser->check_required ("task");

    /* Copy values into output struct */
    parms->output_csv_fn = parser->get_string("output-csv").c_str();
    parms->output_fcsv_fn = parser->get_string("output-fcsv").c_str();
    parms->input_fn = parser->get_string("input").c_str();
    parms->network_fn = parser->get_string("network").c_str();
    if (parser->option("eac")) {
	parms->enforce_anatomic_constraints = true;
    }
    parms->task = parser->get_string("task").c_str();
}

void
do_command_autolabel (int argc, char *argv[])
{
    Autolabel_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    autolabel (&parms);
}
