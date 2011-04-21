/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_convert_h_
#define _itk_image_convert_h_

#include "plm_config.h"
#include "itk_image.h"

template<class T, class U> 
plastimatch1_EXPORT T
plm_image_convert_gpuit_to_itk (Plm_image* pli, T itk_img, U);
template<class T> 
plastimatch1_EXPORT void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, T img);
plastimatch1_EXPORT UCharVecImageType::Pointer
plm_image_convert_itk_uchar_to_itk_uchar_vec (UCharImageType::Pointer im_in);
plastimatch1_EXPORT UCharVecImageType::Pointer
plm_image_convert_gpuit_uint32_to_itk_uchar_vec (Plm_image* pli);
plastimatch1_EXPORT UCharVecImageType::Pointer 
plm_image_convert_itk_uint32_to_itk_uchar_vec (UInt32ImageType::Pointer im_in);

#endif
