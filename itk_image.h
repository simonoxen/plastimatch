/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_h_
#define _itk_image_h_

#include "plm_config.h"
#include <stdio.h>
#include "itkImage.h"
#if (PLM_ITK_ORIENTED_IMAGES)
#include "itkOrientedImage.h"
#endif
#include "itkImageIOBase.h"
#include "plm_image_type.h"
#include "plm_image_patient_position.h"

/* We only deal with 3D images. */
const unsigned int Dimension = 3;

typedef itk::Image < unsigned char, Dimension > UCharImageType;
typedef itk::Image < short, Dimension > ShortImageType;
typedef itk::Image < unsigned short, Dimension > UShortImageType;
#if (CMAKE_SIZEOF_UINT == 4)
typedef itk::Image < int, Dimension > Int32ImageType;
typedef itk::Image < unsigned int, Dimension > UInt32ImageType;
#else
typedef itk::Image < long, Dimension > Int32ImageType;
typedef itk::Image < unsigned long, Dimension > UInt32ImageType;
#endif
typedef itk::Image < float, Dimension > FloatImageType;
typedef itk::Image < double, Dimension > DoubleImageType;

typedef itk::Image < unsigned char, 2 > UCharImage2DType;
typedef itk::Image < short, 2 > ShortImage2DType;
typedef itk::Image < unsigned short, 2 > UShortImage2DType;
#if (CMAKE_SIZEOF_UINT == 4)
typedef itk::Image < int, 2 > Int32Image2DType;
typedef itk::Image < unsigned int, 2 > UInt32Image2DType;
#else
typedef itk::Image < long, 2 > Int32Image2DType;
typedef itk::Image < unsigned long, 2 > UInt32Image2DType;
#endif
typedef itk::Image < float, 2 > FloatImage2DType;
typedef itk::Image < double, 2 > DoubleImage2DType;

typedef itk::Point < float, Dimension > FloatPointType;
typedef itk::Point < double, Dimension > DoublePointType;

typedef itk::Vector < float, Dimension > FloatVectorType;
typedef itk::Vector < double, Dimension > DoubleVectorType;

typedef itk::Image < FloatVectorType, Dimension > DeformationFieldType;

typedef itk::Size < Dimension > SizeType;
typedef itk::Point < double, Dimension >  OriginType;
typedef itk::Vector < double, Dimension > SpacingType;
typedef itk::Matrix<double, Dimension, Dimension> DirectionType;
typedef itk::ImageRegion < Dimension > ImageRegionType;

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT UCharImageType::Pointer itk_image_load_uchar (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT ShortImageType::Pointer itk_image_load_short (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT UShortImageType::Pointer itk_image_load_ushort (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT Int32ImageType::Pointer itk_image_load_int32 (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT UInt32ImageType::Pointer itk_image_load_uint32 (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT FloatImageType::Pointer itk_image_load_float (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT DeformationFieldType::Pointer itk_image_load_float_field (const char* fname);

plastimatch1_EXPORT void itk__GetImageType (std::string fileName,
			itk::ImageIOBase::IOPixelType &pixelType,
			itk::ImageIOBase::IOComponentType &componentType);

template<class T> plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], T image);

template<class T> void itk_image_save (T img_ptr, const char* fname);
template<class T> void itk_image_save_short_dicom (T image, char* dir_name, Plm_image_patient_position);
template<class T> plastimatch1_EXPORT void itk_image_save_uchar (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void itk_image_save_short (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void itk_image_save_ushort (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void itk_image_save_uint32 (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void itk_image_save_float (T img_ptr, char* fname);
#endif
