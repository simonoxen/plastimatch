/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <stdlib.h>

#include "drr_options.h"
#include "pcmd_drr.h"
#include "plm_clp.h"

class Drr_parms {
public:
    std::string input_fn;
    std::string output_fn;
    std::string output_dicom;
};

static void
do_drr (Drr_options *drr_options)
{
}

void
set_image_parms (Drr_options* options)
{
    if (!options->have_image_center) {
	options->image_center[0] = (options->detector_resolution[0]-1)/2.0;
	options->image_center[1] = (options->detector_resolution[1]-1)/2.0;
    }
    if (options->have_image_window) {
	options->image_window[0] = plm_max (0, options->image_window[0]);
	options->image_window[1] = plm_min (options->image_window[1],
            options->detector_resolution[0] - 1);
	options->image_window[2] = plm_max (0, options->image_window[2]);
	options->image_window[3] = plm_min (options->image_window[3],
            options->detector_resolution[1] - 1);
    } else {
	options->image_window[0] = 0;
	options->image_window[1] = options->detector_resolution[0] - 1;
	options->image_window[2] = 0;
	options->image_window[3] = options->detector_resolution[1] - 1;
    }
    options->image_resolution[0] = options->image_window[1]
	- options->image_window[0] + 1;
    options->image_resolution[1] = options->image_window[3]
	- options->image_window[2] + 1;
    if (options->have_angle_diff) {
	options->angle_diff *= (float) (M_TWOPI / 360.0);
    } else {
	options->angle_diff = M_TWOPI / options->num_angles;
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch drr [options] [infile]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Drr_options* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    parser->add_long_option ("A", "threading",
	"Threading option, either \"cpu\" or \"cuda\" (default=cpu)", 1, "cpu");
    parser->add_long_option ("a", "num-images",
	"Generate this many images at equal gantry spacing", 1, "");
    parser->add_long_option ("N", "gantry-angle-spacing",
	"Difference in gantry angle spacing (in degrees)", 1, "");
    parser->add_long_option ("y", "gantry-angle",
	"Gantry angle for image source (in degrees)", 1, "");
    parser->add_long_option ("n", "nrm",
	"Normal vector of detector in format \"x y z\"", 1, "");
    parser->add_long_option ("v", "vup",
	"The vector pointing from the detector center to the top row of "
        "the detector in format \"x y z\"", 1, "");
    parser->add_long_option ("", "sad",
	"The SAD (source-axis-distance) in mm", 1, "");
    parser->add_long_option ("", "sid",
	"The SID (source-image-distance) in mm", 1, "");
    parser->add_long_option ("r", "dim",
	"The output resolution in format \"row col\" (in mm)", 1, "");
    parser->add_long_option ("s", "intensity-scale",
	"Scaling factor for output image intensity", 1, "");
    parser->add_long_option ("e", "exponential",
	"Do exponential mapping of output values", 0);
    parser->add_long_option ("c", "image-center",
	"The image center in the format \"row col\", in pixels", 1, "");
    parser->add_long_option ("z", "detector-size",
	"The physical size of the detector in format \"row col\", in mm",
        1, "");
    parser->add_long_option ("w", "subwindow",
	"Limit DRR output to a subwindow in format \"r1 r2 c1 c2\","
        "in pixels", 1, "");
    parser->add_long_option ("t", "output-format",
	"Select output format, either pgm, pfm, or raw", 1, "");
    parser->add_long_option ("S", "raytrace-details",
	"Create output file with complete ray trace details", 1, "");
    parser->add_long_option ("i", "algorithm",
	"Choose algorithm {exact,uniform}", 1, "");
    parser->add_long_option ("o", "isocenter", 
	"Isocenter position \"x y z\" in DICOM coordinates (mm)", 1, "");
    parser->add_long_option ("G", "geometry-only",
	"Create geometry files only", 0);
    parser->add_long_option ("P", "hu-conversion",
	"Choose HU conversion type {preprocess,inline,none}", 1, "");
    parser->add_long_option ("I", "input", 
	"Input file", 1, "");
    parser->add_long_option ("O", "output",
	"Prefix for output file(s)", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

#if defined (commentout)
    /* Check that an input file was given */
    parser->check_required ("input");
    parser->check_required ("output-img");

    Segment_body *sb = &parms->sb;

    /* Copy values into output struct */
    parms->output_fn = parser->get_string("output-img");
#if defined (commentout)
    parms->output_dicom = parser->get_string("output-dicom");
#endif
    parms->input_fn = parser->get_string("input");
    if (parser->option ("lower-threshold")) {
	sb->m_lower_threshold = parser->get_float("lower-threshold");
    }
#if defined (commentout)
    parms->upper_threshold = parser->get_float("upper-threshold");
#endif
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
do_command_drr (int argc, char *argv[])
{
    Drr_options parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_drr (&parms);
}
