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
#include "rtds.h"
#include "rtss.h"
#include "synthetic_vf.h"
#include "volume_header.h"

typedef struct synthetic_vf_main_parms Synthetic_vf_main_parms;
struct synthetic_vf_main_parms {
    Pstring output_fn;
    Synthetic_vf_parms sv_parms;
};

void
do_synthetic_vf (Synthetic_vf_main_parms *parms)
{
    Synthetic_vf_parms *sv_parms = &parms->sv_parms;

    /* Create image */
    DeformationFieldType::Pointer vf = synthetic_vf (sv_parms);

    /* Save to file */
    itk_image_save (vf, parms->output_fn);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch synth-vf [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Synthetic_vf_main_parms *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char* argv[]
)
{
    Volume_header vh;

    /* Add --help, --version */
    parser->add_default_options ();

    /* Output files */
    parser->add_long_option ("", "output", 
	"output filename", 1, "");
    parser->add_long_option ("", "output-dicom", 
	"output dicom directory", 1, "");
    parser->add_long_option ("", "output-dose-img", 
	"filename for output dose image", 1, "");
    parser->add_long_option ("", "output-ss-img", 
	"filename for output structure set image", 1, "");
    parser->add_long_option ("", "output-type", 
	"data type for output image: {uchar,short,ushort, ulong,float},"
	" default is float", 
	1, "float");

    /* Main pattern */
    parser->add_long_option ("", "pattern",
	"synthetic pattern to create: {"
	"donut, enclosed_rect, gauss, grid, lung, osd, rect, sphere"
	"}, default is gauss", 
	1, "gauss");

    /* Image size */
    parser->add_long_option ("", "origin", 
	"location of first image voxel in mm \"x y z\"", 1, "0 0 0");
    parser->add_long_option ("", "dim", 
	"size of output image in voxels \"x [y z]\"", 1, "100");
    parser->add_long_option ("", "spacing", 
	"voxel spacing in mm \"x [y z]\"", 1, "5");
    parser->add_long_option ("", "direction-cosines", 
	"oriention of x, y, and z axes; Specify either preset value,"
	" {identity,rotated-{1,2,3},sheared},"
	" or 9 digit matrix string \"a b c d e f g h i\"", 1, "");
    parser->add_long_option ("", "volume-size", 
	"size of output image in mm \"x [y z]\"", 1, "500");

    /* Image intensities */
    parser->add_long_option ("", "background", 
	"intensity of background region", 1, "-1000");
    parser->add_long_option ("", "foreground", 
	"intensity of foreground region", 1, "0");

    /* Donut options */
    parser->add_long_option ("", "donut-center", 
	"location of donut center in mm \"x [y z]\"", 1, "0 0 0");
    parser->add_long_option ("", "donut-radius", 
	"size of donut in mm \"x [y z]\"", 1, "50 50 20");
    parser->add_long_option ("", "donut-rings", 
	"number of donut rings (2 rings for traditional donut)", 1, "2");
	
    /* Gaussian options */
    parser->add_long_option ("", "gauss-center", 
	"location of Gaussian center in mm \"x [y z]\"", 1, "0 0 0");
    parser->add_long_option ("", "gauss-std", 
	"width of Gaussian in mm \"x [y z]\"", 1, "100");

    /* Rect options */
    parser->add_long_option ("", "rect-size", 
	"width of rectangle in mm \"x [y z]\","
	" or locations of rectangle corners in mm"
	" \"x1 x2 y1 y2 z1 z2\"", 1, "-50 50 -50 50 -50 50");

    /* Sphere options */
    parser->add_long_option ("", "sphere-center", 
	"location of sphere center in mm \"x y z\"", 1, "0 0 0");
    parser->add_long_option ("", "sphere-radius", 
	"radius of sphere in mm \"x [y z]\"", 1, "50");

    /* Grid pattern options */
    parser->add_long_option ("", "grid-pattern", 
	"grid pattern spacing in voxels \"x [y z]\"", 1, "10");

    /* Lung options */
    parser->add_long_option ("", "lung-tumor-pos", 
	"position of tumor in mm \"z\" or \"x y z\"", 1, "0");

    /* Parse the command line arguments */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option("output") && !parser->option("output-dicom")) {
	throw dlib::error (
	    "Error, you must specify either --output or --output-dicom.\n"
	);
    }

    /* Copy values into output struct */
    Synthetic_vf_parms *sv_parms = &parms->sv_parms;

    /* Basic options */
    parms->output_fn = parser->get_string("output").c_str();

    /* Main pattern */
    std::string arg = parser->get_string ("pattern");
    if (arg == "translation") {
	sv_parms->pattern = Synthetic_vf_parms::PATTERN_TRANSLATION;
    }
    else if (arg == "radial") {
	sv_parms->pattern = Synthetic_vf_parms::PATTERN_RADIAL;
    }
    else if (arg == "zero") {
	sv_parms->pattern = Synthetic_vf_parms::PATTERN_ZERO;
    }
    else {
	throw (dlib::error ("Error. Unknown --pattern argument: " + arg));
    }

    /* Image size */
    parser->assign_plm_long_13 (vh.m_dim, "dim");

    /* Direction cosines */
    if (parser->option ("direction-cosines")) {
	std::string arg = parser->get_string("direction-cosines");
	if (!vh.m_direction_cosines.set_from_string (arg)) {
	    throw (dlib::error ("Error parsing --direction-cosines "
		    "(should have nine numbers)\n"));
	}
    }

    /* If origin not specified, volume is centered about size */
    float volume_size[3];
    parser->assign_float13 (volume_size, "volume-size");
    if (parser->option ("origin")) {
	parser->assign_float13 (vh.m_origin, "origin");
    } else {
	for (int d = 0; d < 3; d++) {
	    /* GCS FIX: This should include direction cosines */
	    vh.m_origin[d] = - 0.5 * volume_size[d] 
		+ 0.5 * volume_size[d] / vh.m_dim[d];
	}
    }

    /* If spacing not specified, set spacing from size and resolution */
    if (parser->option ("spacing")) {
	parser->assign_float13 (vh.m_spacing, "spacing");
    } else {
	for (int d = 0; d < 3; d++) {
	    vh.m_spacing[d] 
		= volume_size[d] / ((float) vh.m_dim[d]);
	}
    }

#if defined (commentout)
    /* Image intensities */
    sv_parms->background = parser->get_float ("background");
    sv_parms->foreground = parser->get_float ("foreground");

    /* Donut options */
    parser->assign_float13 (sv_parms->donut_center, "donut-center");
    parser->assign_float13 (sv_parms->donut_radius, "donut-radius");
    sv_parms->donut_rings = parser->get_int ("donut-rings");

    /* Gaussian options */
    parser->assign_float13 (sv_parms->gauss_center, "gauss-center");
    parser->assign_float13 (sv_parms->gauss_std, "gauss-std");

    /* Rect options */
    int rc = sscanf (parser->get_string("rect-size").c_str(), 
	"%g %g %g %g %g %g", 
	&(sv_parms->rect_size[0]), 
	&(sv_parms->rect_size[1]), 
	&(sv_parms->rect_size[2]), 
	&(sv_parms->rect_size[3]), 
	&(sv_parms->rect_size[4]), 
	&(sv_parms->rect_size[5]));
    if (rc == 1) {
	sv_parms->rect_size[0] = - 0.5 * sv_parms->rect_size[0];
	sv_parms->rect_size[1] = - sv_parms->rect_size[0];
	sv_parms->rect_size[2] = + sv_parms->rect_size[0];
	sv_parms->rect_size[3] = - sv_parms->rect_size[0];
	sv_parms->rect_size[4] = + sv_parms->rect_size[0];
	sv_parms->rect_size[5] = - sv_parms->rect_size[0];
    }
    else if (rc == 3) {
	sv_parms->rect_size[4] = - 0.5 * sv_parms->rect_size[2];
	sv_parms->rect_size[2] = - 0.5 * sv_parms->rect_size[1];
	sv_parms->rect_size[0] = - 0.5 * sv_parms->rect_size[0];
	sv_parms->rect_size[1] = - sv_parms->rect_size[0];
	sv_parms->rect_size[3] = - sv_parms->rect_size[2];
	sv_parms->rect_size[5] = - sv_parms->rect_size[4];
    }
    else if (rc != 6) {
	throw (dlib::error ("Error. Option --rect_size must have "
		"one, three, or six arguments\n"));
    }

    /* Sphere options */
    parser->assign_float13 (sv_parms->sphere_center, "sphere-center");
    parser->assign_float13 (sv_parms->sphere_radius, "sphere-radius");
#endif
}

void
do_command_synth_vf (int argc, char* argv[])
{
    Synthetic_vf_main_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_synthetic_vf (&parms);
}
