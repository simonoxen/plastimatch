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
//#include "synthetic_mha_main.h"
#include "synthetic_mha.h"

typedef dlib::cmd_line_parser<char>::check_1a_c Clp;

typedef struct synthetic_mha_main_parms Synthetic_mha_main_parms;
struct synthetic_mha_main_parms {
    CBString output_fn;
    Synthetic_mha_parms sm_parms;
};

void
do_synthetic_mha (CBString *fn, Synthetic_mha_parms *parms)
{
    /* Create image */
    FloatImageType::Pointer img = synthetic_mha (parms);

    /* Save to file */
    switch (parms->output_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	itk_image_save_uchar (img, (const char*) fn);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	itk_image_save_short (img, (const char*) fn);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	itk_image_save_ushort (img, (const char*) fn);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	itk_image_save_uint32 (img, (const char*) fn);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save_float (img, (const char*) fn);
	break;
    }
}

static void
print_usage (dlib::Plm_clp& parser)
{
    std::cout << "Usage: synthetic_mha [options]\n";
    parser.print_options (std::cout);
    std::cout << std::endl;
}

void
parse_args (Synthetic_mha_main_parms* parms, int argc, char* argv[])
{
    dlib::Plm_clp parser;
    try {
	/* Basic options */
        parser.add_long_option ("", "output", "Output filename (required)", 
	    1, "");
        parser.add_long_option ("", "output-type", 
	    "Data type for output file: {uchar,short,ushort, ulong,float},"
	    " default is float", 
	    1, "float");
	parser.add_long_option ("h", "help", "Display this help message");

	/* Main pattern */
        parser.add_long_option ("", "pattern",
	    "Synthetic pattern to create: {gauss,rect, sphere},"
	    " default is gauss", 
	    1, "gauss");

	/* Image size */
        parser.add_long_option ("", "origin", 
	    "Location of first image voxel in mm \"x y z\"", 1);
        parser.add_long_option ("", "dim", 
	    "Size of output image in voxels \"x [y z]\"", 1);
        parser.add_long_option ("", "spacing", 
	    "Voxel spacing in mm \"x [y z]\"", 1);
        parser.add_long_option ("", "volume-size", 
	    "Size of output image in mm \"x [y z]\"", 1);

	/* Image intensities */
        parser.add_long_option ("", "background", 
	    "Intensity of background region", 1);
        parser.add_long_option ("", "foreground", 
	    "Intensity of foreground region", 1);
	
	/* Gaussian options */
        parser.add_long_option ("", "gauss-center", 
	    "Location of Gaussian center in mm \"x y z\"", 1);
        parser.add_long_option ("", "gauss-std", 
	    "Width of Gaussian in mm \"x [y z]\"", 1);

	/* Rect options */
        parser.add_long_option ("", "rect-size", 
	    "Width of rectangle in mm \"x [y z]\","
	    " or locations of rectangle corners in mm"
	    " \"x1 x2 y1 y2 z1 z2\"", 1);

	/* Sphere options */
        parser.add_long_option ("", "sphere-center", 
	    "Location of sphere center in mm \"x y z\"", 1);
        parser.add_long_option ("", "sphere-radius", 
	    "Radius of sphere in mm \"x [y z]\"", 1);

	/* Parse the command line arguments */
        parser.parse (argc,argv);
    }
    catch (std::exception& e) {
        /* Catch cmd_line_parse_error exceptions and print usage message. */
	std::cout << e.what() << std::endl;
	print_usage (parser);
	exit (1);
    }
    catch (...) {
	std::cout << "Some error occurred" << std::endl;
    }

    /* Check if the -h option was given */
    if (parser.option("h") || parser.option("help")) {
	print_usage (parser);
	exit (0);
    }

    if (parser.option("background")) {
	printf ("Background is %s\n", 
	    parser.option("background").argument().c_str());
    }

    /* Check that an output file was given */
    if (!parser.option("output")) {
	std::cout << "Error, you must specify the --output option.\n";
	print_usage (parser);
	exit (1);
    }

    /* Copy values into output struct */
    Synthetic_mha_parms *sm_parms = &parms->sm_parms;
    parms->output_fn = parser.get_value("output").c_str();
    sm_parms->background = dlib::sa = parser.get_value("background");
    sm_parms->foreground = dlib::sa = parser.get_value("foreground");
}

int 
main (int argc, char* argv[])
{
    Synthetic_mha_main_parms parms;

    //parse_args_old (&parms, argc, argv);

    parse_args (&parms, argc, argv);

    do_synthetic_mha (&parms.output_fn, &parms.sm_parms);

    return 0;
}
