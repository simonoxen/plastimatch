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

/* We only deal with these kinds of images. */
enum PlmImageType {
    PLM_IMG_TYPE_UNDEFINED, 
    PLM_IMG_TYPE_ITK_CHAR, 
    PLM_IMG_TYPE_ITK_UCHAR, 
    PLM_IMG_TYPE_ITK_SHORT, 
    PLM_IMG_TYPE_ITK_USHORT, 
    PLM_IMG_TYPE_ITK_LONG, 
    PLM_IMG_TYPE_ITK_ULONG, 
    PLM_IMG_TYPE_ITK_FLOAT, 
    PLM_IMG_TYPE_ITK_DOUBLE, 
    PLM_IMG_TYPE_ITK_FLOAT_FIELD, 
    PLM_IMG_TYPE_GPUIT_FLOAT, 
    PLM_IMG_TYPE_GPUIT_FLOAT_FIELD, 
};

/* We only deal with 3D images. */
const unsigned int Dimension = 3;

typedef itk::Image < unsigned char, Dimension > UCharImageType;
typedef itk::Image < short, Dimension > ShortImageType;
typedef itk::Image < unsigned short, Dimension > UShortImageType;
#if (CMAKE_SIZEOF_UINT == 4)
typedef itk::Image < unsigned int, Dimension > UInt32ImageType;
#else
typedef itk::Image < unsigned long, Dimension > UInt32ImageType;
#endif
typedef itk::Image < float, Dimension > FloatImageType;
typedef itk::Image < double, Dimension > DoubleImageType;

typedef itk::Image < unsigned char, 2 > UCharImage2DType;
typedef itk::Image < short, 2 > ShortImage2DType;
typedef itk::Image < unsigned short, 2 > UShortImage2DType;
#if (CMAKE_SIZEOF_UINT == 4)
typedef itk::Image < unsigned int, 2 > UInt32Image2DType;
#else
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
plastimatch1_EXPORT UCharImageType::Pointer load_uchar (char* fname, PlmImageType* original_type);
plastimatch1_EXPORT ShortImageType::Pointer load_short (char* fname, PlmImageType* original_type);
plastimatch1_EXPORT UShortImageType::Pointer load_ushort (char* fname, PlmImageType* original_type);
plastimatch1_EXPORT UInt32ImageType::Pointer load_uint32 (char* fname, PlmImageType* original_type);
plastimatch1_EXPORT FloatImageType::Pointer load_float (char* fname, PlmImageType* original_type);
plastimatch1_EXPORT DeformationFieldType::Pointer load_float_field (char* fname);

plastimatch1_EXPORT void itk__GetImageType (std::string fileName,
			itk::ImageIOBase::IOPixelType &pixelType,
			itk::ImageIOBase::IOComponentType &componentType);

template<class T> plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], T image);

template<class T> void save_image (T img_ptr, char* fname);
template<class T> void save_short_dicom (T image, char* dir_name);
template<class T> plastimatch1_EXPORT void save_uchar (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void save_short (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void save_ushort (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void save_ulong (T img_ptr, char* fname);
template<class T> plastimatch1_EXPORT void save_float (T img_ptr, char* fname);
#endif
