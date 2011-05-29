/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>

#include "bstring_util.h"
#include "pcmd_xf_convert.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "xform_convert.h"

class Xf_convert_parms {
public:
    CBString input_fn;
    CBString output_fn;
    CBString output_type;

    Xform_convert xfc;
};

void
set_output_xform_type (Xform_convert *xfc, const CBString& output_type)
{
    if (output_type == "vf") {
	xfc->xf_out_type = XFORM_ITK_VECTOR_FIELD;
    }
    else if (output_type == "bspline") {
	xfc->xf_out_type = XFORM_GPUIT_BSPLINE;
    }
    else if (output_type == "itk_bsp" || output_type == "itk_bspline") {
	xfc->xf_out_type = XFORM_ITK_BSPLINE;
    }
    else {
	print_and_exit ("Sorry, can't convert output type\n");
    }
}

static void
do_xf_convert (Xf_convert_parms *parms)
{
    Xform_convert *xfc = &parms->xfc;

    /* Set up inputs */
    xfc->xf_in = new Xform;
    xfc->xf_out = new Xform;
    xform_load (xfc->xf_in, parms->input_fn);
    set_output_xform_type (xfc, parms->output_type);
    
    /* Do conversion */
    printf ("about to xform_convert\n");
    xform_convert (xfc);
    printf ("did xform_convert\n");

    /* Save output file */
    xform_save (xfc->xf_out, parms->output_fn);
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch xf-convert [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Xf_convert_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    parser->add_long_option ("h", "help", "Display this help message");

    /* Basic options */
    parser->add_long_option ("", "input", 
	"Input xform filename (required)", 1, "");
    parser->add_long_option ("", "output", 
	"Output xform filename (required)", 1, "");
    parser->add_long_option ("", "output-type", 
	"Type of xform to create (required), choose from "
	"{bspline, itk_bspline, vf}", 1, "");

    parser->add_long_option ("", "origin", 
	"Location of first image voxel in mm \"x y z\"", 1, "");
    parser->add_long_option ("", "dim", 
	"Size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "spacing", 
	"Voxel spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "grid-spacing", 
	"B-spline grid spacing in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "nobulk", 
	"Omit bulk transform for itk_bspline", 0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Check if the -h option was given */
    parser->check_help ();

    /* Check that an input file was given */
    parser->check_required ("input");
    parser->check_required ("output");
    parser->check_required ("output-type");

    Xform_convert *xfc = &parms->xfc;

    /* Copy values into output struct */
    parms->output_fn = parser->get_string("output").c_str();
    parms->input_fn = parser->get_string("input").c_str();
    if (parser->option ("output-type")) {
	parms->output_type = parser->get_string("output-type").c_str();
    }

    /* Geometry options */
    if (parser->option ("origin")) {
	parser->assign_float13 (xfc->origin, "origin");
    }
    if (parser->option ("spacing")) {
	parser->assign_float13 (xfc->spacing, "spacing");
    }
    if (parser->option ("dim")) {
	parser->assign_int13 (xfc->dim, "dim");
    }
    if (parser->option ("grid-spacing")) {
	parser->assign_float13 (xfc->grid_spac, "grid-spacing");
    }

    if (parser->option ("nobulk")) {
	xfc->nobulk = true;
    }
}

void
do_command_xf_convert (int argc, char *argv[])
{
    Xf_convert_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_xf_convert (&parms);
}
