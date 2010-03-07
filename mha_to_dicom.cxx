/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This program is used to convert from mha to dicom */
#include <time.h>
#include "plm_config.h"
#include "itkImage.h"
#include "itk_image.h"

int
main(int argc, char *argv[])
{
    if (argc != 3 && argc != 4) {
	std::cerr << "Usage: " << argv[0];
	std::cerr << " input_filename output_dir [output_type]" << std::endl;
	std::cerr << " output_type can be: uchar, short, ushort, or float" << std::endl;
	return 1;
    }

    int output_type = 0;
    // input file name
    char* input_fn = argv[1];
    // output to dicom directory
    char* dicom_dir = argv[2];
    
    if (argc == 4) {
	if (0 == strcmp(argv[3], "uchar")) {
	    output_type = PLM_IMG_TYPE_ITK_UCHAR;
	} else if (0 == strcmp(argv[3], "short")) {
	    output_type = PLM_IMG_TYPE_ITK_SHORT;
	} else if (0 == strcmp(argv[3], "ushort")) {
	    output_type = PLM_IMG_TYPE_ITK_USHORT;
	} else if (0 == strcmp(argv[3] , "float")) {
	    output_type = PLM_IMG_TYPE_ITK_FLOAT;
	} else {
	    fprintf (stderr, "Error.  Unknown output type.\n");
	    exit (1);
	}
    } else {
        output_type = PLM_IMG_TYPE_ITK_SHORT;
    }

    if (output_type == PLM_IMG_TYPE_ITK_UCHAR) {
        UCharImageType::Pointer input_image = itk_image_load_uchar (
	    input_fn, 0);
        itk_image_save_short_dicom (input_image, dicom_dir);
    } else if (output_type == PLM_IMG_TYPE_ITK_SHORT) {
        ShortImageType::Pointer input_image = itk_image_load_short (
	    input_fn, 0);
        itk_image_save_short_dicom (input_image, dicom_dir);
    } else if (output_type == PLM_IMG_TYPE_ITK_USHORT) {
        UShortImageType::Pointer input_image = itk_image_load_ushort (
	    input_fn, 0);
        itk_image_save_short_dicom (input_image, dicom_dir);
    } else if (output_type == PLM_IMG_TYPE_ITK_FLOAT) {
        FloatImageType::Pointer input_image = itk_image_load_float (
	    input_fn, 0);
        itk_image_save_short_dicom (input_image, dicom_dir);
    }

    printf ("Finished!\n");
    return 0;
}
