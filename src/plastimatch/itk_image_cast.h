/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_cast_h_
#define _itk_image_cast_h_

#include "plm_config.h"
#include "itk_image.h"

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
template<class T> plastimatch1_EXPORT UCharImageType::Pointer cast_uchar (T image);
template<class T> plastimatch1_EXPORT ShortImageType::Pointer cast_short (T image);
template<class T> plastimatch1_EXPORT UShortImageType::Pointer cast_ushort (T image);
template<class T> plastimatch1_EXPORT Int32ImageType::Pointer cast_int32 (T image);
template<class T> plastimatch1_EXPORT UInt32ImageType::Pointer cast_uint32 (T image);
template<class T> plastimatch1_EXPORT FloatImageType::Pointer cast_float (T image);
template<class T> plastimatch1_EXPORT DoubleImageType::Pointer cast_double (T image);
#endif
