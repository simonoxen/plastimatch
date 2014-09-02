/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_save_h_
#define _itk_image_save_h_

#include "plmbase_config.h"
#include "itk_image_type.h"

class Metadata;
class Rt_study_metadata;
class Slice_index;

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
template<class T> void itk_image_save (T img_ptr, const char* fname);
template<class T> void itk_image_save (T img_ptr, const std::string& fname);
template<class T> void itk_image_save_short_dicom (T image, const char* dir_name, Rt_study_metadata *);

template<class T> PLMBASE_API void itk_image_save_char (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_uchar (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_short (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_ushort (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_int32 (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_uint32 (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_float (T img_ptr, const char* fname);
template<class T> PLMBASE_API void itk_image_save_double (T img_ptr, const char* fname);
#endif
