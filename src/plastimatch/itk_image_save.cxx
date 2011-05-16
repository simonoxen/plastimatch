/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"

#include "plm_int.h"
#include "itk_image.h"
#include "itk_dicom_save.h"
#include "itk_image_cast.h"
#include "file_util.h"
#include "print_and_exit.h"
#include "logfile.h"
#include "plm_image_patient_position.h"

#if (defined(_WIN32) || defined(WIN32))
#define snprintf _snprintf
#define mkdir(a,b) _mkdir(a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

/* -----------------------------------------------------------------------
   Writing image files
   ----------------------------------------------------------------------- */
template<class T> 
void
itk_image_save (T image, const char* fname)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageFileWriter< ImageType >  WriterType;

    logfile_printf ("Trying to write image to %s\n", fname);

    typename WriterType::Pointer writer = WriterType::New();
    writer->SetInput (image);
    writer->SetFileName (fname);
    make_directory_recursive (fname);
    if (extension_is (fname, "nrrd")) {
	writer->SetUseCompression (true);
    }
    try {
	writer->Update();
    }
    catch (itk::ExceptionObject& excp) {
	printf ("ITK exception writing image file.\n");
	std::cout << excp << std::endl;
    }
}

template<class T> 
void
itk_image_save_uchar (T image, const char* fname)
{
    UCharImageType::Pointer uchar_img = cast_uchar (image);
    itk_image_save (uchar_img, fname);
}

template<class T> 
void
itk_image_save_short (T image, const char* fname)
{
    ShortImageType::Pointer short_img = cast_short (image);
    itk_image_save (short_img, fname);
}

template<class T> 
void
itk_image_save_ushort (T image, const char* fname)
{
    UShortImageType::Pointer ushort_img = cast_ushort (image);
    itk_image_save (ushort_img, fname);
}

template<class T> 
void
itk_image_save_short_dicom (
    T image, 
    const char* dir_name, 
    Referenced_dicom_dir *rdd, 
    const Img_metadata *img_metadata, 
    Plm_image_patient_position patient_pos)
{
    ShortImageType::Pointer short_img = cast_short (image);
    itk_dicom_save (short_img, dir_name, rdd, img_metadata, patient_pos);
}

template<class T> 
void
itk_image_save_uint32 (T image, const char* fname)
{
    UInt32ImageType::Pointer uint32_img = cast_uint32 (image);
    itk_image_save (uint32_img, fname);
}

template<class T> 
void
itk_image_save_float (T image, const char* fname)
{
    FloatImageType::Pointer float_img = cast_float (image);
    itk_image_save (float_img, fname);
}

template<class T> 
void
itk_image_save_double (T image, const char* fname)
{
    DoubleImageType::Pointer double_img = cast_double (image);
    itk_image_save (double_img, fname);
}

/* Explicit instantiations */
template plastimatch1_EXPORT void itk_image_save(UCharImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(ShortImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(UShortImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(Int32ImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(UInt32ImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(DoubleImageType::Pointer, const char*);

template plastimatch1_EXPORT void itk_image_save(DeformationFieldType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(UCharImage4DType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(UCharVecImageType::Pointer, const char*);

template plastimatch1_EXPORT void itk_image_save_uchar (FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save_short (FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save_ushort (FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save_uint32 (FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save_float (FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save_double (FloatImageType::Pointer, const char*);

template plastimatch1_EXPORT void itk_image_save_short_dicom (UCharImageType::Pointer, const char*, Referenced_dicom_dir *, const Img_metadata *img_metadata, Plm_image_patient_position);
template plastimatch1_EXPORT void itk_image_save_short_dicom (ShortImageType::Pointer, const char*, Referenced_dicom_dir *, const Img_metadata *img_metadata, Plm_image_patient_position);
template plastimatch1_EXPORT void itk_image_save_short_dicom (UShortImageType::Pointer, const char*, Referenced_dicom_dir *, const Img_metadata *img_metadata, Plm_image_patient_position);
template plastimatch1_EXPORT void itk_image_save_short_dicom (UInt32ImageType::Pointer, const char*, Referenced_dicom_dir *, const Img_metadata *img_metadata, Plm_image_patient_position);
template plastimatch1_EXPORT void itk_image_save_short_dicom (FloatImageType::Pointer, const char*, Referenced_dicom_dir *, const Img_metadata *img_metadata, Plm_image_patient_position);

