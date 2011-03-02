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
    Synthetic_vf_parms sm_parms;
};

void
do_synthetic_vf (CBString& fn, Synthetic_vf_parms *parms)
{
    /* Create image */
    FloatImageType::Pointer img = synthetic_vf (parms);

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
    std::cout << "Usage: synthetic_vf [options]\n";
    parser.print_options (std::cout);
    std::cout << std::endl;
}

void
parse_args (Synthetic_vf_main_parms* parms, int argc, char* argv[])
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
	    "Location of first image voxel in mm \"x y z\"", 1, "0 0 0");
        parser.add_long_option ("", "dim", 
	    "Size of output image in voxels \"x [y z]\"", 1, "100");
        parser.add_long_option ("", "spacing", 
	    "Voxel spacing in mm \"x [y z]\"", 1, "5");
        parser.add_long_option ("", "volume-size", 
	    "Size of output image in mm \"x [y z]\"", 1, "500");

	/* Image intensities */
        parser.add_long_option ("", "background", 
	    "Intensity of background region", 1, "-1000");
        parser.add_long_option ("", "foreground", 
	    "Intensity of foreground region", 1, "0");
	
	/* Gaussian options */
        parser.add_long_option ("", "gauss-center", 
	    "Location of Gaussian center in mm \"x [y z]\"", 1, "0 0 0");
        parser.add_long_option ("", "gauss-std", 
	    "Width of Gaussian in mm \"x [y z]\"", 1, "100");

	/* Rect options */
        parser.add_long_option ("", "rect-size", 
	    "Width of rectangle in mm \"x [y z]\","
	    " or locations of rectangle corners in mm"
	    " \"x1 x2 y1 y2 z1 z2\"", 1, "-50 50 -50 50 -50 50");

	/* Sphere options */
        parser.add_long_option ("", "sphere-center", 
	    "Location of sphere center in mm \"x y z\"", 1, "0 0 0");
        parser.add_long_option ("", "sphere-radius", 
	    "Radius of sphere in mm \"x [y z]\"", 1, "50");

	/* Parse the command line arguments */
        parser.parse (argc,argv);

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

	/* Copy values into output struct */
	Synthetic_vf_parms *sm_parms = &parms->sm_parms;

	/* Basic options */
	parms->output_fn = parser.get_string("output").c_str();
	sm_parms->output_type = plm_image_type_parse (
	    parser.get_string("output-type").c_str());

	/* Main pattern */
	std::string arg = parser.get_string ("pattern");
	if (arg == "gauss") {
	    sm_parms->pattern = PATTERN_GAUSS;
	}
	else if (arg == "rect") {
	    sm_parms->pattern = PATTERN_RECT;
	}
	else if (arg == "sphere") {
	    sm_parms->pattern = PATTERN_SPHERE;
	}
	else {
	    throw (dlib::error ("Error. Unknown --pattern argument: " + arg));
	}

	/* Image size */
	parser.assign_int13 (sm_parms->dim, "dim");

	/* If origin not specified, volume is centered about size */
	float volume_size[3];
	parser.assign_float13 (volume_size, "volume-size");
	if (parser.option ("origin")) {
	    parser.assign_float13 (sm_parms->origin, "origin");
	} else {
	    for (int d = 0; d < 3; d++) {
		sm_parms->origin[d] = - 0.5 * volume_size[d] 
		    + 0.5 * volume_size[d] / sm_parms->dim[d];
	    }
	}

	/* If spacing not specified, set spacing from size and resolution */
	if (parser.option ("spacing")) {
	    parser.assign_float13 (sm_parms->spacing, "spacing");
	} else {
	    for (int d = 0; d < 3; d++) {
		sm_parms->spacing[d] 
		    = volume_size[d] / ((float) sm_parms->dim[d]);
	    }
	}

	/* Image intensities */
	sm_parms->background = parser.get_float ("background");
	sm_parms->foreground = parser.get_float ("foreground");

	/* Gaussian options */
	parser.assign_float13 (sm_parms->gauss_center, "gauss-center");
	parser.assign_float13 (sm_parms->gauss_std, "gauss-std");

	/* Rect options */
	int rc = sscanf (parser.get_string("rect-size").c_str(), 
	    "%g %g %g %g %g %g", 
	    &(sm_parms->rect_size[0]), 
	    &(sm_parms->rect_size[1]), 
	    &(sm_parms->rect_size[2]), 
	    &(sm_parms->rect_size[3]), 
	    &(sm_parms->rect_size[4]), 
	    &(sm_parms->rect_size[5]));
	if (rc == 1) {
	    sm_parms->rect_size[0] = - 0.5 * sm_parms->rect_size[0];
	    sm_parms->rect_size[1] = - sm_parms->rect_size[0];
	    sm_parms->rect_size[2] = + sm_parms->rect_size[0];
	    sm_parms->rect_size[3] = - sm_parms->rect_size[0];
	    sm_parms->rect_size[4] = + sm_parms->rect_size[0];
	    sm_parms->rect_size[5] = - sm_parms->rect_size[0];
	}
	else if (rc == 3) {
	    sm_parms->rect_size[4] = - 0.5 * sm_parms->rect_size[2];
	    sm_parms->rect_size[2] = - 0.5 * sm_parms->rect_size[1];
	    sm_parms->rect_size[0] = - 0.5 * sm_parms->rect_size[0];
	    sm_parms->rect_size[1] = - sm_parms->rect_size[0];
	    sm_parms->rect_size[3] = - sm_parms->rect_size[2];
	    sm_parms->rect_size[5] = - sm_parms->rect_size[4];
	}
	else if (rc != 6) {
	    throw (dlib::error ("Error. Option --rect_size must have "
		    "one, three, or six arguments\n"));
	}

	/* Sphere options */
	parser.assign_float13 (sm_parms->sphere_center, "sphere-center");
	parser.assign_float13 (sm_parms->sphere_radius, "sphere-radius");
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

    do_synthetic_vf (parms.output_fn, &parms.sm_parms);

    return 0;
}
