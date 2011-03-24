/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include "itkImageRegionIterator.h"

#include "bstring_util.h"
#include "pcmd_segment.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "resample_mha.h"
#include "segment_body.h"

class Segment_parms {
public:
    CBString input_fn;
    CBString output_fn;
    CBString output_dicom;
    float lower_threshold;
    float upper_threshold;
};

static void
do_segment (Segment_parms *parms)
{
#if defined (commentout)
    Segment_body sb;

    /* Load the input image */
    sb.img_in.load_native (args_info->input_arg);

    /* Set other parameter(s) */
    sb.bot_given = args_info->bot_given;
    sb.bot = args_info->bot_arg;

    /* Do segmentation */
    sb.do_segmentation ();

    /* Save output file */
    sb.img_out.save_image (args_info->output_arg);
#endif
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch segment [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Segment_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    parser->add_long_option ("h", "help", "Display this help message");

    /* Basic options */
    parser->add_long_option ("", "output-img", 
	"Output image filename", 1, "");
    parser->add_long_option ("", "output-dicom", 
	"Output dicom directory (for RTSTRUCT)", 1, "");
    parser->add_long_option ("", "input", 
	"Input image filename (required)", 1, "");
    parser->add_long_option ("", "lower-threshold", 
	"Lower threshold (include voxels above this value)", 1, "");
    parser->add_long_option ("", "upper-threshold", 
	"Upper threshold (include voxels below this value)", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Check if the -h option was given */
    parser->check_help ();

    /* Check that an input file was given */
    parser->check_required ("input");

    /* Copy values into output struct */
    parms->output_fn = parser->get_string("output-img").c_str();
    parms->output_dicom = parser->get_string("output-dicom").c_str();
    parms->input_fn = parser->get_string("input").c_str();
    parms->lower_threshold = parser->get_float("lower-threshold");
    parms->upper_threshold = parser->get_float("upper-threshold");
}

void
do_command_segment (int argc, char *argv[])
{
    Segment_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_segment (&parms);
}
