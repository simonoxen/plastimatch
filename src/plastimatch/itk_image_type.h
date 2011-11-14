/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_type_h_
#define _itk_image_type_h_

#include "plm_config.h"
#include "itkImage.h"
#include "itkImageIOBase.h"
#if (PLM_ITK_ORIENTED_IMAGES)
#include "itkOrientedImage.h"
#endif
#include "itkVectorImage.h"

#include "itk_point.h"
#include "plm_int.h"

/* 4D images */
typedef itk::Image < unsigned char, 4 > UCharImage4DType;

/* 3D images */
typedef itk::Image < int8_t, 3 > CharImageType;
typedef itk::Image < uint8_t, 3 > UCharImageType;
typedef itk::Image < int16_t, 3 > ShortImageType;
typedef itk::Image < uint16_t, 3 > UShortImageType;
typedef itk::Image < int32_t, 3 > Int32ImageType;
typedef itk::Image < uint32_t, 3 > UInt32ImageType;
typedef itk::Image < float, 3 > FloatImageType;
typedef itk::Image < double, 3 > DoubleImageType;

typedef itk::VectorImage < uint8_t, 3 > UCharVecImageType;

/* 2D images */
typedef itk::Image < int8_t, 2 > CharImage2DType;
typedef itk::Image < uint8_t, 2 > UCharImage2DType;
typedef itk::Image < int16_t, 2 > ShortImage2DType;
typedef itk::Image < uint16_t, 2 > UShortImage2DType;
typedef itk::Image < int32_t, 2 > Int32Image2DType;
typedef itk::Image < uint32_t, 2 > UInt32Image2DType;
typedef itk::Image < float, 2 > FloatImage2DType;
typedef itk::Image < double, 2 > DoubleImage2DType;

typedef itk::VectorImage < uint8_t, 2 > UCharVecImage2DType;

/* Vector field */
typedef itk::Image < FloatVector3DType, 3 > DeformationFieldType;

#endif
