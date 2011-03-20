/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_img_extract_h_
#define _ss_img_extract_h_

#include "plm_config.h"
#include "itk_image.h"

UCharImageType::Pointer
ss_img_extract_bit (UInt32ImageType::Pointer image, unsigned int bit);
UCharImageType::Pointer
ss_img_extract_bit (UCharVecImageType::Pointer image, unsigned int bit);
UCharImageType::Pointer
ss_img_extract_uchar (
    UCharVecImageType::Pointer im_in, 
    unsigned int uchar_no
);
void
ss_img_insert_uchar (
    UCharVecImageType::Pointer vec_img, 
    UCharImageType::Pointer uchar_img, 
    unsigned int uchar_no
);

#endif
