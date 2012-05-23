/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_h_
#define _itk_image_h_

#include "plmbase_config.h"
#include "sys/plm_int.h"

#include "itk_image_type.h"

class Plm_image_header;
class Volume_header;

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
PLMBASE_C_API void itk_image_get_props (
    const std::string& fileName,
    int *num_dimensions, 
    itk::ImageIOBase::IOPixelType *pixel_type, 
    itk::ImageIOBase::IOComponentType *component_type, 
    int *num_components
);

template<class T> PLMBASE_API void get_image_header (plm_long dim[3], float offset[3], float spacing[3], T image);
template<class T> PLMBASE_API void itk_image_get_volume_header (Volume_header *vh, T image);
template<class T> PLMBASE_API void itk_image_set_header (T image, const Plm_image_header *pih);
template<class T> PLMBASE_API void itk_image_set_header (T image, const Plm_image_header& pih);
template<class T, class U> PLMBASE_API void itk_image_header_copy (T dest, U src);
#endif
