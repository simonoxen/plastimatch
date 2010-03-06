/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_img_extract_h_
#define _ss_img_extract_h_

#include "plm_config.h"
#include "itk_image.h"

UCharImageType::Pointer
ss_img_extract (UInt32ImageType::Pointer image, int bit);

#endif
