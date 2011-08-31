/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>

#include "pstring.h"
#include "itk_image.h"
#include "itk_image_load.h"
#include "pcmd_probe.h"
#include "plm_clp.h"
#include "plm_file_format.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "pointset.h"
#include "pstring.h"

class Probe_parms {
public:
    Pstring input_fn;
    Pstring index_string;

public:
    Probe_parms () {
    }
};

static void
parse_float_string (const Pstring& ps)
{
    const char* p = (const char*) ps;
    int rc = 0;
    int n;

    do {
	float f[3];

	n = 0;
	rc = sscanf (p, "%f %f %f;%n", &f[0], &f[1], &f[2], &n);
	p += n;
    } while (rc >= 3 && n > 0);
}

static void
probe_img_main (Probe_parms *parms)
{
    FloatImageType::Pointer img = itk_image_load_float (
	(const char*) parms->input_fn, 0);
    FloatImageType::RegionType rg = img->GetLargestPossibleRegion ();

    parse_float_string (parms->index_string);

    FloatImageType::IndexType pixel_index;
    pixel_index[0] = 0;
    pixel_index[1] = 0;
    pixel_index[2] = 0;

    FloatImageType::PixelType pixel_value = img->GetPixel (pixel_index);
}

static void
probe_vf_main (Probe_parms *parms)
{
}

static void
do_probe (Probe_parms *parms)
{
    switch (plm_file_format_deduce ((const char*) parms->input_fn)) {
    case PLM_FILE_FMT_VF:
	probe_vf_main (parms);
	break;
    case PLM_FILE_FMT_IMG:
    default:
	probe_img_main (parms);
	break;
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch probe [options] file\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Probe_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("i", "index", 
	"List of voxel indices, such as \"i j k;i j k;...\"", 1, "");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an index was given */
    parser->check_required ("index");

    /* Check that an input file was given */
    if (parser->number_of_arguments() != 1) {
	std::string extra_arg = (*parser)[1];
	throw (dlib::error ("Error.  Unknown option " + extra_arg));
    }

    /* Copy values into output struct */
    parms->input_fn = (*parser)[0].c_str();
    parms->index_string = parser->get_string("index").c_str();
}

void
do_command_probe (int argc, char *argv[])
{
    Probe_parms parms;

    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_probe (&parms);
}
