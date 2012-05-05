/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_img_extract_h_
#define _ss_img_extract_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

class Plm_image;

API UCharImageType::Pointer ss_img_extract_bit (
        Plm_image *image,
        unsigned int bit
);
API UCharImageType::Pointer ss_img_extract_bit (
        UInt32ImageType::Pointer image,
        unsigned int bit
);
API UCharImageType::Pointer ss_img_extract_bit (
        UCharVecImageType::Pointer image,
        unsigned int bit
);
template<class T> API typename itk::Image<typename T::ObjectType::IOPixelType, T::ObjectType::ImageDimension>::Pointer 
ss_img_extract_uchar (T im_in, unsigned int uchar_no);

API void ss_img_insert_uchar (
        UCharVecImageType::Pointer vec_img,
        UCharImageType::Pointer uchar_img,
        unsigned int uchar_no
);

#endif
