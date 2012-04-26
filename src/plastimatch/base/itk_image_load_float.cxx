/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "itk_dicom_load.h"
#include "itk_image.h"
#include "itk_image_load.h"
#include "itk_image_load.txx"

FloatImageType::Pointer
itk_image_load_float (const char* fname, Plm_image_type* original_type)
{
    FloatImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_float (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<float>(0));
    }
    //return orient_image (img);
    return img;
}
