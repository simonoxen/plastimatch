/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>

#include "plmsys.h"

#include "getopt.h"
#include "itk_crop.h"
#include "pcmd_crop.h"
#include "plm_image.h"

static void
crop_main (Crop_Parms* parms)
{
    Plm_image plm_image;

    plm_image.load_native ((const char*) parms->img_in_fn);

    switch (plm_image.m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	plm_image.m_itk_uchar 
	    = itk_crop (plm_image.m_itk_uchar, parms->crop_vox);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	plm_image.m_itk_short 
		= itk_crop (plm_image.m_itk_short, parms->crop_vox);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	plm_image.m_itk_uint32 
		= itk_crop (plm_image.m_itk_uint32, parms->crop_vox);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	plm_image.m_itk_float 
		= itk_crop (plm_image.m_itk_float, parms->crop_vox);
	break;
    default:
	print_and_exit ("Unhandled image type in resample_main()\n");
	break;
    }

    plm_image.convert_and_save (
	(const char*) parms->img_out_fn, 
	plm_image.m_type);
}

static void
crop_print_usage (void)
{
    printf ("Usage: plastimatch crop [options]\n"
	    "Required:\n"
	    "    --input=image_in\n"
	    "    --output=image_out\n"
	    "    --voxels=\"x-min x-max y-min y-max z-min z-max\" (integers)\n"
	    );
    exit (-1);
}

static void
crop_parse_args (Crop_Parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "input",          required_argument,      NULL,           2 },
	{ "output",         required_argument,      NULL,           3 },
	{ "voxels",         required_argument,      NULL,           4 },
	{ NULL,             0,                      NULL,           0 }
    };

    while ((ch = getopt_long(argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 2:
	    parms->img_in_fn = optarg;
	    break;
	case 3:
	    parms->img_out_fn = optarg;
	    break;
	case 4:
	    if (sscanf (optarg, 
		    "%d %d %d %d %d %d", 
		    &parms->crop_vox[0],
		    &parms->crop_vox[1],
		    &parms->crop_vox[2],
		    &parms->crop_vox[3],
		    &parms->crop_vox[4],
		    &parms->crop_vox[5]) != 6)
	    {
		printf ("Error: voxels takes 6 arguments\n");
		crop_print_usage ();
	    }
	    break;
	default:
	    break;
	}
    }
    if (parms->img_in_fn.length() == 0 || parms->img_out_fn.length() == 0) {
	printf ("Error: must specify --input and --output\n");
	crop_print_usage ();
    }
}

void
do_command_crop (int argc, char *argv[])
{
    Crop_Parms parms;
    
    crop_parse_args (&parms, argc, argv);

    crop_main (&parms);

    printf ("Finished!\n");
}
