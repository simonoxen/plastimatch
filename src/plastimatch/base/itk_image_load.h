/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_load_h_
#define _itk_image_load_h_

#include "plmbase_config.h"
#include "itk_image_type.h"
#include "plm_image_type.h"

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT CharImageType::Pointer itk_image_load_char (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT UCharImageType::Pointer itk_image_load_uchar (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT ShortImageType::Pointer itk_image_load_short (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT UShortImageType::Pointer itk_image_load_ushort (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT Int32ImageType::Pointer itk_image_load_int32 (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT UInt32ImageType::Pointer itk_image_load_uint32 (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT FloatImageType::Pointer itk_image_load_float (const char* fname, Plm_image_type* original_type);
plastimatch1_EXPORT DoubleImageType::Pointer itk_image_load_double (const char* fname, Plm_image_type* original_type);

plastimatch1_EXPORT DeformationFieldType::Pointer itk_image_load_float_field (const char* fname);
plastimatch1_EXPORT UCharVecImageType::Pointer itk_image_load_uchar_vec (const char* fname);

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

#endif
