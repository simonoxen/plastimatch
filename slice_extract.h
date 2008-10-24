/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _slice_extract_h
#define _slice_extract_h

#include "itkImage.h"

template<class T>
typename itk::Image<T,2>::Pointer slice_extract (typename itk::Image<T,3>::Pointer reader, int index, T);

#endif
