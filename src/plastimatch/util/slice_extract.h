/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _slice_extract_h
#define _slice_extract_h

#include "plmutil_config.h"
#include "itkImage.h"

template<class T> API
typename itk::Image<typename T::ObjectType::PixelType,2>::Pointer slice_extract (T in_img, int slice_no);

API UCharVecImage2DType::Pointer slice_extract (UCharVecImageType::Pointer, int);

#endif
