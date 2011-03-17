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
#include "itk_dicom_load.h"
#include "itk_image.h"
#include "itk_image_cast.h"
#include "itk_image_load.h"
#include "itk_image_load.txx"
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

UCharImageType::Pointer
itk_image_load_uchar (const char* fname, Plm_image_type* original_type)
{
    UCharImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uchar (fname);
    } else {
	img = itk_image_load_any (fname, original_type, 
	    static_cast<unsigned char>(0));
    }
    return orient_image (img);
}

ShortImageType::Pointer
itk_image_load_short (const char* fname, Plm_image_type* original_type)
{
    ShortImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_short (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<short>(0));
    }
    return orient_image (img);
}

UShortImageType::Pointer
itk_image_load_ushort (const char* fname, Plm_image_type* original_type)
{
    UShortImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_ushort (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<unsigned short>(0));
    }
    return orient_image (img);
}

Int32ImageType::Pointer
itk_image_load_int32 (const char* fname, Plm_image_type* original_type)
{
    Int32ImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_int32 (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<int32_t>(0));
    }
    return orient_image (img);
}

UInt32ImageType::Pointer
itk_image_load_uint32 (const char* fname, Plm_image_type* original_type)
{
    UInt32ImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uint32 (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<uint32_t>(0));
    }
    return orient_image (img);
}

FloatImageType::Pointer
itk_image_load_float (const char* fname, Plm_image_type* original_type)
{
    FloatImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_float (fname);
    } else {
	img = itk_image_load_any (fname, original_type, static_cast<float>(0));
    }
    return orient_image (img);
}

UCharVecImageType::Pointer
itk_image_load_uchar_vec (const char* fname)
{
    typedef itk::ImageFileReader< UCharVecImageType > ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName (fname);

    try {
	reader->Update();
    }
    catch (itk::ExceptionObject& excp) {
	std::cerr << "ITK exception reading file." << std::endl;
	std::cerr << excp << std::endl;
	return 0;
    }
    UCharVecImageType::Pointer img = reader->GetOutput();
    return orient_image (img);
}

DeformationFieldType::Pointer
itk_image_load_float_field (const char* fname)
{
    DeformationFieldType::Pointer img 
	= itk_image_load<DeformationFieldType> (fname);
    return orient_image (img);
}
