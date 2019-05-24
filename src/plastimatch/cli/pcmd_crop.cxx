/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include <string>

#include "itk_crop.h"
#include "pcmd_crop.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "print_and_exit.h"

class Crop_parms {
public:
    std::string img_in_fn;
    std::string img_out_fn;
    bool have_coordinates;
    int crop_vox[6];
    float crop_coords[6];
public:
    Crop_parms () {
        img_in_fn = "";
        img_out_fn = "";
        for (int i = 0; i < 6; i++) {
            crop_vox[i] = 0;
            crop_coords[i] = 0;
        }
        have_coordinates = false;
    }
};

static void
crop_main (Crop_parms* parms)
{
    Plm_image plm_image;

    plm_image.load_native (parms->img_in_fn);

    switch (plm_image.m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
        if (parms->have_coordinates) {
            plm_image.m_itk_uchar 
                = itk_crop_by_coord (plm_image.m_itk_uchar, parms->crop_coords);
        } else {
            plm_image.m_itk_uchar 
                = itk_crop_by_index (plm_image.m_itk_uchar, parms->crop_vox);
        }
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
        if (parms->have_coordinates) {
            plm_image.m_itk_short 
		= itk_crop_by_coord (plm_image.m_itk_short, parms->crop_coords);
        } else {
            plm_image.m_itk_short 
		= itk_crop_by_index (plm_image.m_itk_short, parms->crop_vox);
        }
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
        if (parms->have_coordinates) {
            plm_image.m_itk_uint32 
		= itk_crop_by_coord (plm_image.m_itk_uint32, parms->crop_coords);
        } else {
            plm_image.m_itk_uint32 
		= itk_crop_by_index (plm_image.m_itk_uint32, parms->crop_vox);
        }
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
        if (parms->have_coordinates) {
            plm_image.m_itk_float 
		= itk_crop_by_coord (plm_image.m_itk_float, parms->crop_coords);
        } else {
            plm_image.m_itk_float 
		= itk_crop_by_index (plm_image.m_itk_float, parms->crop_vox);
        }
	break;
    default:
	print_and_exit ("Unhandled image type in resample_main()\n");
	break;
    }

    plm_image.convert_and_save (parms->img_out_fn, plm_image.m_type);
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
    Crop_parms* parms, 
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

    /* Adjustment string */
    parser->add_long_option ("", "voxels", 
        "a string that specifies the voxel indices of the six corners "
        "of the region to be cropped, in the form "
        "\"x1 x2 y1 y2 z1 z2\"", 
        1, "");

    /* Adjustment string */
    parser->add_long_option ("", "coordinates", 
        "a string that specifies the coordinates of the six corners "
        "of the region to be cropped, in the form "
        "\"x1 x2 y1 y2 z1 z2\"", 
        1, "");

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

    /* Check that an output file was given */
    if (!parser->option ("voxels") && !parser->option("coordinates")) {
	throw (dlib::error ("Error.  Please specify the crop region "
		"using the --voxels or --coordinates option"));
    }

    /* Input files */
    parms->img_in_fn = parser->get_string("input").c_str();

    /* Output files */
    parms->img_out_fn = parser->get_string("output").c_str();

    /* Voxels option */
    if (parser->option("voxels")) {
        parser->assign_int_6 (parms->crop_vox, "voxels");
    }
    
    /* Coordinates option */
    if (parser->option("coordinates")) {
        parms->have_coordinates = true;
        parser->assign_float_6 (parms->crop_coords, "coordinates");
    }
}

void
do_command_crop (int argc, char *argv[])
{
    Crop_parms parms;
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    crop_main (&parms);
}
