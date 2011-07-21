/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "itk_image.h"
#include "itk_image_save.h"
#include "math_util.h"
#include "plm_clp.h"
#include "pstring.h"
#include "synthetic_vf.h"

typedef struct synthetic_vf_main_parms Synthetic_vf_main_parms;
struct synthetic_vf_main_parms {
    Pstring output_fn;
    Pstring fixed_fn;
    Synthetic_vf_parms sv_parms;
};

void
do_synthetic_vf (Pstring& fn, Synthetic_vf_parms *parms)
{
#if defined (commentout)
    /* Create vf */
    FloatImageType::Pointer img = synthetic_vf (parms);
#endif
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: synthetic_vf [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
    std::cout << "At least one of the --xf-*** options is required\n";
}

static void
parse_fn (
    Synthetic_vf_main_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* ------------------------------------------------------------
       Create options
       ------------------------------------------------------------ */
    /* Basic options */
    parser->add_long_option ("", "output", "Output filename (required)", 
	1, "");
    parser->add_long_option ("h", "help", "Display this help message");

    /* Image size */
    parser->add_long_option ("", "origin", 
	"Location of first image voxel in mm \"x y z\"", 1, "0 0 0");
    parser->add_long_option ("", "dim", 
	"Size of output image in voxels \"x [y z]\"", 1, "100");
    parser->add_long_option ("", "spacing", 
	"Voxel spacing in mm \"x [y z]\"", 1, "5");
    parser->add_long_option ("", "fixed", 
	"An input image used to set the size of the output ", 1, "");

    /* Patterns */
    parser->add_long_option ("", "xf-radial",
	"Radial expansion (or contraction).", 1);
    parser->add_long_option ("", "xf-trans",
	"Uniform translation in mm \"x y z\"", 1);
    parser->add_long_option ("", "xf-zero", "Null transform");

    /* ------------------------------------------------------------
       Parse options
       ------------------------------------------------------------ */
    parser->parse (argc, argv);

    /* Check if the -h option was given */
    parser->check_help ();

    /* Check that an output file was given */
    parser->check_required ("output");

    /* Check that a xf option was given */
    if (!parser->option("xf-radial") && 
	!parser->option("xf-trans") && 
	!parser->option("xf-zero"))
    {
	std::cout << 
	    "Error, you must specify one of the --xf-*** options.\n";
	usage_fn (parser, argc, argv);
	exit (1);
    }

    /* Copy values into output struct */
    Synthetic_vf_parms *sv_parms = &parms->sv_parms;

    /* Basic options */
    parms->output_fn = parser->get_string("output").c_str();

    /* Image geometry */
    parser->assign_int13 (sv_parms->dim, "dim");
    parser->assign_float13 (sv_parms->origin, "origin");
    parser->assign_float13 (sv_parms->spacing, "spacing");
    if (parser->option ("fixed")) {
	parms->output_fn = parser->get_string("fixed").c_str();
    }

    /* Patterns */
    if (parser->option("xf-trans")) {
	sv_parms->pattern = Synthetic_vf_parms::PATTERN_TRANS;
    } else if (parser->option("xf-zero")) {
	sv_parms->pattern = Synthetic_vf_parms::PATTERN_ZERO;
    } else {
	throw (dlib::error ("Error. Unknown --xf argument."));
    }
}

int 
main (int argc, char* argv[])
{
    Synthetic_vf_main_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv);

    do_synthetic_vf (parms.output_fn, &parms.sv_parms);

    return 0;
}
