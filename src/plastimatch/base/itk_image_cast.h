/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_cast_h_
#define _itk_image_cast_h_

#include "plmbase_config.h"
#include "itk_image_type.h"

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
template<class T> API CharImageType::Pointer cast_char (T image);
template<class T> API UCharImageType::Pointer cast_uchar (T image);
template<class T> API ShortImageType::Pointer cast_short (T image);
template<class T> API UShortImageType::Pointer cast_ushort (T image);
template<class T> API Int32ImageType::Pointer cast_int32 (T image);
template<class T> API UInt32ImageType::Pointer cast_uint32 (T image);
template<class T> API FloatImageType::Pointer cast_float (T image);
template<class T> API DoubleImageType::Pointer cast_double (T image);
#endif
