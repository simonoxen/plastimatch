/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "plmbase.h"

#include "itk_image_load.txx"

UCharImageType::Pointer
itk_image_load_uchar (const char* fname, Plm_image_type* original_type)
{
    UCharImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uchar (fname);
    } else {
	img = itk_image_load_any (fname, original_type, 
	    static_cast<unsigned char>(0));
    }
    //return orient_image (img);
    return img;
}
