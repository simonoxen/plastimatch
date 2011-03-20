/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _slice_extract_h
#define _slice_extract_h

#include "plm_config.h"
#include "itkImage.h"

template<class T> plastimatch1_EXPORT
typename itk::Image<typename T::ObjectType::PixelType,2>::Pointer slice_extract (T in_img, int slice_no);

#endif
