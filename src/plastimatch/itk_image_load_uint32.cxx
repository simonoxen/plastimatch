/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itk_dicom_load.h"
#include "itk_image.h"
#include "itk_image_load.h"
#include "itk_image_load.txx"

UInt32ImageType::Pointer
itk_image_load_uint32 (const char* fname, Plm_image_type* original_type)
{
    UInt32ImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uint32 (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<uint32_t>(0));
    }
    return orient_image (img);
}
