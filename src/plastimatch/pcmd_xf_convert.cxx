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

    /* Geometry options */
    bool m_have_dim;
    bool m_have_origin;
    bool m_have_spacing;
    Volume_header m_vh;

    bool m_have_grid_spacing;
    float m_grid_spacing[3];

    Xform_convert xfc;
public:
    Xf_convert_parms () {
	m_have_dim = false;
	m_have_origin = false;
	m_have_spacing = false;
	m_have_grid_spacing = false;
    }
};

void
set_output_xform_type (Xform_convert *xfc, const CBString& output_type)
{
    if (output_type == "vf") {
	xfc->m_xf_out_type = XFORM_ITK_VECTOR_FIELD;
    }
    else if (output_type == "bspline") {
	xfc->m_xf_out_type = XFORM_GPUIT_BSPLINE;
    }
    else if (output_type == "itk_bsp" || output_type == "itk_bspline") {
	xfc->m_xf_out_type = XFORM_ITK_BSPLINE;
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
    xfc->m_xf_in = new Xform;
    xfc->m_xf_out = new Xform;
    xform_load (xfc->m_xf_in, parms->input_fn);
    set_output_xform_type (xfc, parms->output_type);

    /* Set grid spacing as needed */
    xfc->m_xf_in->get_grid_spacing (xfc->m_grid_spac);
    if (parms->m_have_grid_spacing) {
	for (int d = 0; d < 3; d++) {
	    xfc->m_grid_spac[d] = parms->m_grid_spacing[d];
	}
    }
    if (xfc->m_xf_in->m_type == XFORM_GPUIT_BSPLINE) {
	Bspline_xform* bxf = xfc->m_xf_in->get_gpuit_bsp();
	printf ("vox_per_rgn = %d %d %d\n", 
	    bxf->vox_per_rgn[0],
	    bxf->vox_per_rgn[1],
	    bxf->vox_per_rgn[2]
	);
	printf ("grid_spac = %g %g %g\n", 
	    bxf->grid_spac[0],
	    bxf->grid_spac[1],
	    bxf->grid_spac[2]
	);
	printf ("grid_spac = %g %g %g\n", 
	    xfc->m_grid_spac[0],
	    xfc->m_grid_spac[1],
	    xfc->m_grid_spac[2]
	);
    }

    /* Set volume header as needed */
    xfc->m_xf_in->get_volume_header (&xfc->m_volume_header);
    if (parms->m_have_dim) {
	xfc->m_volume_header.set_dim (parms->m_vh.m_dim);
    }
    if (parms->m_have_origin) {
	xfc->m_volume_header.set_origin (parms->m_vh.m_origin);
    }
    if (parms->m_have_spacing) {
	xfc->m_volume_header.set_spacing (parms->m_vh.m_spacing);
    }
    
    /* Do conversion */
    printf ("about to xform_convert\n");
    xform_convert (xfc);
    printf ("did xform_convert\n");

    /* Save output file */
    xform_save (xfc->m_xf_out, parms->output_fn);
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

    parser->add_long_option ("", "dim", 
	"Size of output image in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "origin", 
	"Location of first image voxel in mm \"x y z\"", 1, "");
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
    if (parser->option ("dim")) {
	parms->m_have_dim = true;
	parser->assign_int13 (parms->m_vh.m_dim, "dim");
    }
    if (parser->option ("origin")) {
	parms->m_have_origin = true;
	parser->assign_float13 (parms->m_vh.m_origin, "origin");
    }
    if (parser->option ("spacing")) {
	parms->m_have_spacing = true;
	parser->assign_float13 (parms->m_vh.m_spacing, "spacing");
    }
    if (parser->option ("grid-spacing")) {
	parms->m_have_grid_spacing = true;
	parser->assign_float13 (parms->m_grid_spacing, "grid-spacing");
    }

    if (parser->option ("nobulk")) {
	xfc->m_nobulk = true;
    }
}

void
do_command_xf_convert (int argc, char *argv[])
{
    Xf_convert_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_xf_convert (&parms);
}
