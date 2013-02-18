/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <list>

#include "image_boundary.h"
#include "itk_image_load.h"
#include "itk_image_save.h"
#include "plm_clp.h"
#include "pstring.h"

class Boundary_parms {
public:
    Pstring output_fn;
    Pstring input_fn;
};

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] input_file\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Boundary_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "output", 
        "filename for output image", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Check that one, and only one, input file was given */
    if (parser->number_of_arguments() == 0) {
	throw (dlib::error ("Error.  You must specify an input file"));
	
    } else if (parser->number_of_arguments() > 1) {
	std::string extra_arg = (*parser)[1];
	throw (dlib::error ("Error.  Unknown option " + extra_arg));
    }

    /* Copy input filenames to parms struct */
    parms->input_fn = (*parser)[0].c_str();

    /* Output files */
    parms->output_fn = parser->get_string("output").c_str();
}

void
do_command_boundary (int argc, char *argv[])
{
    Boundary_parms parms;

    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    UCharImageType::Pointer input_image = itk_image_load_uchar (
        parms.input_fn.c_str(), 0);
    UCharImageType::Pointer output_image = do_image_boundary (input_image);
    itk_image_save (output_image, parms.output_fn.c_str());
}
