/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "gamma_dose_comparison.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_math.h"
#include "pcmd_gamma.h"
#include "print_and_exit.h"

static void
gamma_main (Gamma_parms* parms)
{
    Gamma_dose_comparison gdc;

    gdc.set_reference_image (parms->ref_image_fn.c_str());
    gdc.set_compare_image (parms->cmp_image_fn.c_str());

    gdc.set_spatial_tolerance (parms->dta_tolerance);
    gdc.set_dose_difference_tolerance (parms->dose_tolerance);

    gdc.run ();

    Plm_image::Pointer gamma_image = gdc.get_gamma_image ();
    gamma_image->save_image (parms->out_image_fn);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] image_1 image_2\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Gamma_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "output", "output image", 1, "");

    /* Gamma options */
    parser->add_long_option ("", "dose-tolerance", 
        "the scaling coefficient for dose difference in percent "
        "(default is .03)", 1, ".03");
    parser->add_long_option ("", "dta-tolerance", 
        "the distance-to-agreement (DTA) scaling coefficient in mm "
        "(default is 3)", 1, "3");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Check that two input files were given */
    if (parser->number_of_arguments() < 2) {
	throw (dlib::error ("Error.  You must specify two input files"));
	
    } else if (parser->number_of_arguments() > 2) {
	std::string extra_arg = (*parser)[1];
	throw (dlib::error ("Error.  Extra argument " + extra_arg));
    }

    /* Input files */
    parms->ref_image_fn = (*parser)[0].c_str();
    parms->cmp_image_fn = (*parser)[1].c_str();

    /* Output files */
    parms->out_image_fn = parser->get_string("output").c_str();

    /* Gamma options */
    parms->dose_tolerance = parser->get_float("dose-tolerance");
    parms->dta_tolerance = parser->get_float("dta-tolerance");
}

void
do_command_gamma (int argc, char *argv[])
{
    Gamma_parms parms;

    printf ("Hello world.\n");
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    printf ("Running gamma_main.\n");
    gamma_main (&parms);
}
