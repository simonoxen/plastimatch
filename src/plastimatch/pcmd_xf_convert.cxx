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

    Xform_convert xfc;
};

static void
do_xf_convert (Xf_convert_parms *parms)
{
    Xform_convert *xfc = &parms->xfc;

#if defined (commentout)
    /* Load the input image */
    sb->img_in.load_native (parms->input_fn);

    /* Do segmentation */
    sb->do_segmentation ();

    /* Save output file */
    sb->img_out.save_image (parms->output_fn);
#endif
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
    parser->add_long_option ("", "output-img", 
	"Output image filename", 1, "");
#if defined (commentout)
    parser->add_long_option ("", "output-dicom", 
	"Output dicom directory (for RTSTRUCT)", 1, "");
#endif
    parser->add_long_option ("", "input", 
	"Input image filename (required)", 1, "");
    parser->add_long_option ("", "bottom", 
	"Bottom of patient (top of couch)", 1, "");
#if defined (commentout)
    parser->add_long_option ("", "lower-threshold", 
	"Lower threshold (include voxels above this value)", 1, "");
    parser->add_long_option ("", "upper-threshold", 
	"Upper threshold (include voxels below this value)", 1, "");
#endif
    parser->add_long_option ("", "debug", "Create debug images", 0);
    parser->add_long_option ("", "fast", "Use reduced image size", 0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Check if the -h option was given */
    parser->check_help ();

    /* Check that an input file was given */
    parser->check_required ("input");
    parser->check_required ("output-img");

    Xform_convert *xfc = &parms->xfc;

#if defined (commentout)
    /* Copy values into output struct */
    parms->output_fn = parser->get_string("output-img").c_str();
    parms->input_fn = parser->get_string("input").c_str();
    if (parser->option ("bottom")) {
	sb->m_bot_given = true;
	sb->m_bot = parser->get_float ("bottom");
    }
    if (parser->option ("fast")) {
	sb->m_fast = true;
    }
    if (parser->option ("debug")) {
	sb->m_debug = true;
    }
#endif
}

void
do_command_xf_convert (int argc, char *argv[])
{
    Xf_convert_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_xf_convert (&parms);
}
