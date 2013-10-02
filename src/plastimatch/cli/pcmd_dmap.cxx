/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <string>

#include "distance_map.h"
#include "itk_image_save.h"
#include "pcmd_dmap.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "print_and_exit.h"

class Dmap_parms {
public:
    std::string img_in_fn;
    std::string img_out_fn;
    std::string algorithm;
    bool squared_distance;
    bool inside_positive;
public:
    Dmap_parms () {
        squared_distance = false;
        inside_positive = false;
    }
};

static void
dmap_main (Dmap_parms* parms)
{
    Distance_map dmap;
    dmap.set_input_image (parms->img_in_fn);
    if (parms->algorithm == "maurer") {
        dmap.set_algorithm (Distance_map::ITK_SIGNED_MAURER);
    }
    else if (parms->algorithm == "danielsson") {
        dmap.set_algorithm (Distance_map::ITK_SIGNED_DANIELSSON);
    }
    else {
        print_and_exit ("Error.  Unknown algorithm option: %s",
            parms->algorithm.c_str());
    }

    dmap.set_inside_is_positive (parms->inside_positive);
    dmap.set_use_squared_distance (parms->squared_distance);

    dmap.run ();
    FloatImageType::Pointer dmap_image = dmap.get_output_image();
    itk_image_save (dmap_image, parms->img_out_fn.c_str());
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options]\n", argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Dmap_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Input files */
    parser->add_long_option ("", "input", 
        "input directory or filename", 1, "");

    /* Output files */
    parser->add_long_option ("", "output", 
        "output image", 1, "");

    /* Algorithm options */
    parser->add_long_option ("", "algorithm", 
        "a string that specifies the algorithm used for distance "
        "map calculation, either \"maurer\" or \"danielsson\" "
        "(default is \"maurer\")",
        1, "maurer");
    parser->add_long_option ("", "squared-distance",
        "return the squared distance instead of distance", 0);
    parser->add_long_option ("", "inside-positive",
        "voxels inside the structure should be positive"
        " (by default they are negative)", 0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an output file was given */
    if (!parser->option ("input")) {
	throw (dlib::error ("Error.  Please specify an input file "
		"using the --input option"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output")) {
	throw (dlib::error ("Error.  Please specify an output file "
		"using the --output option"));
    }

    /* Input files */
    parms->img_in_fn = parser->get_string("input");

    /* Output files */
    parms->img_out_fn = parser->get_string("output");

    /* Algorithm options */
    parms->algorithm = parser->get_string("algorithm");
    if (parser->option("squared-distance")) {
        parms->squared_distance = true;
    }
    if (parser->option("inside-positive")) {
        parms->inside_positive = true;
    }
}

void
do_command_dmap (int argc, char *argv[])
{
    Dmap_parms parms;
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    dmap_main (&parms);
}
