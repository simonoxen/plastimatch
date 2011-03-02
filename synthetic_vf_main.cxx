/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "bstring_util.h"
#include "itk_image.h"
#include "itk_image_save.h"
#include "math_util.h"
#include "plm_dlib_clp.h"
#include "synthetic_vf.h"

typedef dlib::cmd_line_parser<char>::check_1a_c Clp;

typedef struct synthetic_vf_main_parms Synthetic_vf_main_parms;
struct synthetic_vf_main_parms {
    CBString output_fn;
    CBString fixed_fn;
    Synthetic_vf_parms sv_parms;
};

void
do_synthetic_vf (CBString& fn, Synthetic_vf_parms *parms)
{
#if defined (commentout)
    /* Create vf */
    FloatImageType::Pointer img = synthetic_vf (parms);
#endif
}

static void
print_usage (dlib::Plm_clp& parser)
{
    std::cout << "Usage: synthetic_vf [options]\n";
    parser.print_options (std::cout);
    std::cout << std::endl;
    std::cout << "At least one of the --xf-*** options is required\n";
}

void
parse_args (Synthetic_vf_main_parms* parms, int argc, char* argv[])
{
    dlib::Plm_clp parser;
    try {
	/* ------------------------------------------------------------
	   Create options
	   ------------------------------------------------------------ */
	/* Basic options */
        parser.add_long_option ("", "output", "Output filename (required)", 
	    1, "");
	parser.add_long_option ("h", "help", "Display this help message");

	/* Image size */
        parser.add_long_option ("", "origin", 
	    "Location of first image voxel in mm \"x y z\"", 1, "0 0 0");
        parser.add_long_option ("", "dim", 
	    "Size of output image in voxels \"x [y z]\"", 1, "100");
        parser.add_long_option ("", "spacing", 
	    "Voxel spacing in mm \"x [y z]\"", 1, "5");
        parser.add_long_option ("", "fixed", 
	    "An input image used to set the size of the output ", 1, "");

	/* Patterns */
        parser.add_long_option ("", "xf-trans",
	    "Uniform translation in mm \"x y z\"", 1);
        parser.add_long_option ("", "xf-zero", "Null transform");

	/* ------------------------------------------------------------
	   Parse options
	   ------------------------------------------------------------ */
        parser.parse (argc,argv);

	/* ------------------------------------------------------------
	   Check options and copy into struct
	   ------------------------------------------------------------ */
	/* Check if the -h option was given */
	if (parser.option("h") || parser.option("help")) {
	    print_usage (parser);
	    exit (0);
	}

	/* Check that an output file was given */
	if (!parser.option("output")) {
	    std::cout << "Error, you must specify the --output option.\n";
	    print_usage (parser);
	    exit (1);
	}

	/* Check that a xf option was given */
	if (!parser.option("xf-trans") && 
	    !parser.option("xf-zero"))
	{
	    std::cout << 
		"Error, you must specify one of the --xf-*** options.\n";
	    print_usage (parser);
	    exit (1);
	}

	/* Copy values into output struct */
	Synthetic_vf_parms *sv_parms = &parms->sv_parms;

	/* Basic options */
	parms->output_fn = parser.get_string("output").c_str();

	/* Image geometry */
	parser.assign_int13 (sv_parms->dim, "dim");
	parser.assign_float13 (sv_parms->origin, "origin");
	parser.assign_float13 (sv_parms->spacing, "spacing");
	if (parser.option ("fixed")) {
	    parms->output_fn = parser.get_string("fixed").c_str();
	}

	/* Patterns */
	if (parser.option("xf-trans")) {
	    sv_parms->pattern = Synthetic_vf_parms::PATTERN_TRANS;
	} else if (parser.option("xf-zero")) {
	    sv_parms->pattern = Synthetic_vf_parms::PATTERN_ZERO;
	} else {
	    throw (dlib::error ("Error. Unknown --xf argument."));
	}
    }
    catch (std::exception& e) {
        /* Catch cmd_line_parse_error exceptions and print usage message. */
	std::cout << e.what() << std::endl;
	print_usage (parser);
	exit (1);
    }
    catch (...) {
	std::cerr << "An unspecified error occurred.\n";
	exit (1);
    }
}

int 
main (int argc, char* argv[])
{
    Synthetic_vf_main_parms parms;

    parse_args (&parms, argc, argv);

    do_synthetic_vf (parms.output_fn, &parms.sv_parms);

    return 0;
}
