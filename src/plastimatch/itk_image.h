/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_h_
#define _itk_image_h_

#include "plm_config.h"
#include <stdio.h>
#include "itkImage.h"
#include "itkImageIOBase.h"
#if (PLM_ITK_ORIENTED_IMAGES)
#include "itkOrientedImage.h"
#endif
#include "itkVectorImage.h"

#include "itk_point.h"
#include "plm_image_type.h"

class Plm_image_header;
class Volume_header;

/* 4D images */
typedef itk::Image < unsigned char, 4 > UCharImage4DType;

/* 3D images */
typedef itk::Image < unsigned char, 3 > UCharImageType;
typedef itk::Image < short, 3 > ShortImageType;
typedef itk::Image < unsigned short, 3 > UShortImageType;
#if (CMAKE_SIZEOF_UINT == 4)
typedef itk::Image < int, 3 > Int32ImageType;
typedef itk::Image < unsigned int, 3 > UInt32ImageType;
#else
typedef itk::Image < long, 3 > Int32ImageType;
typedef itk::Image < unsigned long, 3 > UInt32ImageType;
#endif
typedef itk::Image < float, 3 > FloatImageType;
typedef itk::Image < double, 3 > DoubleImageType;

typedef itk::VectorImage < unsigned char, 3 > UCharVecImageType;

/* 2D images */
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

typedef itk::VectorImage < unsigned char, 2 > UCharVecImage2DType;

/* Vector field */
typedef itk::Image < FloatVector3DType, 3 > DeformationFieldType;

/* Other types */
typedef itk::VariableLengthVector<unsigned char> UCharVecType;
typedef itk::Size < 3 > SizeType;
typedef itk::Point < double, 3 >  OriginType;
typedef itk::Vector < double, 3 > SpacingType;
typedef itk::Matrix < double, 3, 3 > DirectionType;
typedef itk::ImageRegion < 3 > ImageRegionType;

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT 
void
itk_image_get_props (
    std::string fileName,
    int *num_dimensions, 
    itk::ImageIOBase::IOPixelType &pixel_type, 
    itk::ImageIOBase::IOComponentType &component_type, 
    int *num_components
);

template<class T> plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], T image);
template<class T> plastimatch1_EXPORT void itk_image_get_volume_header (Volume_header *vh, T image);
template<class T> plastimatch1_EXPORT void itk_image_set_header (T image, Plm_image_header *pih);
template<class T, class U> plastimatch1_EXPORT void itk_image_header_copy (T dest, U src);
#endif
