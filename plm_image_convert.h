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

template<class U>
plastimatch1_EXPORT UCharImage4DType::Pointer
plm_image_convert_gpuit_to_itk_uchar_4d (Plm_image* pli, U);

template<class T> 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, T img);

template<class T> plastimatch1_EXPORT UCharImage4DType::Pointer plm_image_convert_itk_to_itk_uchar_4d (T);

#endif
