/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "dice_statistics.h"
#include "hausdorff_statistics.h"
#include "itk_image_load.h"
#include "itk_resample.h"
#include "plm_clp.h"
#include "plm_image_header.h"
#include "pstring.h"

class Pcmd_dice_parms {
public:
    Pstring reference_image_fn;
    Pstring test_image_fn;
public:
    Pcmd_dice_parms () { }
};

/* For differing resolutions, resamples image_2 to image_1 */
void check_resolution (
    UCharImageType::Pointer *image_1,
    UCharImageType::Pointer *image_2
)
{
    if ((*image_1)->GetLargestPossibleRegion().GetSize() !=
        (*image_2)->GetLargestPossibleRegion().GetSize())
    {
        Plm_image_header pih;
        pih.set_from_itk_image (*image_1);
        *image_2 = resample_image (*image_2, &pih, 0, false);
    }
}

static void
usage_fn (dlib::Plm_clp *parser, int argc, char *argv[])
{
    std::cout << 
        "Usage: plastimatch dice [options] reference-image test-image\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Pcmd_dice_parms *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char *argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that two input files were given */
    if (parser->number_of_arguments() < 2) {
	throw (dlib::error ("Error.  You must specify two input files"));
	
    } else if (parser->number_of_arguments() > 2) {
	std::string extra_arg = (*parser)[1];
	throw (dlib::error ("Error.  Extra argument " + extra_arg));
    }

    /* Copy values into output struct */
    parms->reference_image_fn = (*parser)[0].c_str();
    parms->test_image_fn = (*parser)[1].c_str();
}

void
do_command_dice (int argc, char *argv[])
{
    Pcmd_dice_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    UCharImageType::Pointer image_1 = itk_image_load_uchar (
        parms.reference_image_fn.c_str(), 0);
    UCharImageType::Pointer image_2 = itk_image_load_uchar (
        parms.test_image_fn.c_str(), 0);

    check_resolution (&image_1, &image_2);

    do_dice<unsigned char> (image_1, image_2, stdout);
    do_hausdorff<unsigned char> (image_1, image_2);
    do_contour_mean_dist<unsigned char> (image_1, image_2);
}
