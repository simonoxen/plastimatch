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
    Functions
   ----------------------------------------------------------------------- */
// This function is copied and modified from Slicer3 (itkPluginUtilities.h)
void
itk_image_get_props (
    std::string fileName,
    int *num_dimensions, 
    itk::ImageIOBase::IOPixelType &pixel_type, 
    itk::ImageIOBase::IOComponentType &component_type, 
    int *num_components
)
{
    pixel_type = itk::ImageIOBase::UNKNOWNPIXELTYPE;
    component_type = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE;
    *num_dimensions = 0;
    typedef itk::Image<short, 3> ImageType;
    itk::ImageFileReader<ImageType>::Pointer imageReader =
	itk::ImageFileReader<ImageType>::New();
    imageReader->SetFileName(fileName.c_str());
    try {
	imageReader->UpdateOutputInformation();
	pixel_type = imageReader->GetImageIO()->GetPixelType();
	component_type = imageReader->GetImageIO()->GetComponentType();
	*num_dimensions = imageReader->GetImageIO()->GetNumberOfDimensions();
	*num_components = imageReader->GetImageIO()->GetNumberOfComponents();
    } catch (itk::ExceptionObject &ex) {
	printf ("ITK exception.\n");
	std::cout << ex << std::endl;
    }
}

template<class RdrT>
void
itk_image_load_rdr (RdrT reader, const char *fn)
{
    reader->SetFileName(fn);
    try {
	reader->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("ITK exception reading image file: %s!\n",fn);
	std::cout << ex << std::endl;
	getchar();
	exit(1);
    }
}

template<class T>
typename T::Pointer
itk_image_load (const char *fn)
{

    typedef typename itk::ImageFileReader < T > ReaderType;
    typename ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(fn);
    try {
	reader->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("ITK exception reading image file: %s!\n",fn);
	std::cout << ex << std::endl;
	getchar();
	exit(1);
    }
    typename T::Pointer img = reader->GetOutput();
    return img;
}

/* -----------------------------------------------------------------------
   Orienting Images
   ----------------------------------------------------------------------- */
template<class T>
T
orient_image (T img)
{
    typedef typename T::ObjectType ImageType;
    typedef typename itk::OrientImageFilter<ImageType,ImageType> OrienterType;
    
    typename OrienterType::Pointer orienter = OrienterType::New();
    orienter->UseImageDirectionOn ();
    orienter->SetDesiredCoordinateOrientation (itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI);
    orienter->SetInput (img);
    orienter->Update ();
    return orienter->GetOutput ();
}

/* -----------------------------------------------------------------------
   Reading Image Headers
   ----------------------------------------------------------------------- */
template<class T>
void
get_image_header (int dim[3], float offset[3], float spacing[3], T image)
{
    typename T::ObjectType::RegionType rg = image->GetLargestPossibleRegion ();
    typename T::ObjectType::PointType og = image->GetOrigin();
    typename T::ObjectType::SpacingType sp = image->GetSpacing();
    typename T::ObjectType::SizeType sz = rg.GetSize();

    /* Copy header & allocate data for gpuit float */
    for (int d = 0; d < 3; d++) {
	dim[d] = sz[d];
	offset[d] = og[d];
	spacing[d] = sp[d];
    }
}

/* -----------------------------------------------------------------------
   Reading image files
   ----------------------------------------------------------------------- */
template<class T, class U>
static
typename itk::Image< U, 3 >::Pointer
load_any_2 (const char* fname, T, U)
{
    typedef typename itk::Image < T, 3 > TImageType;
    typedef typename itk::Image < U, 3 > UImageType;
    typedef itk::ImageFileReader < TImageType > TReaderType;
    typedef typename itk::CastImageFilter < 
		TImageType, UImageType > CastFilterType;

    /* Load image as type T */
    typename TReaderType::Pointer rdr = TReaderType::New();
    itk_image_load_rdr (rdr, fname);
    typename TImageType::Pointer input_image = rdr->GetOutput();

    /* Convert images to type U */
    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput (input_image);
    typename UImageType::Pointer image = caster->GetOutput();
    image->Update();

    /* Return type U */
    return image;
}

static void
set_original_type (Plm_image_type *original_type,
		   Plm_image_type t)
{
    if (original_type) {
	*original_type = t;
    }
}

template<class U>
typename itk::Image< U, 3 >::Pointer
itk_image_load_any (
    const char* fname,
    Plm_image_type* original_type, 
    U otype)
{
    if (!file_exists (fname)) {
	print_and_exit ("Can't open file \"%s\" for read\n", fname);
    }

    int num_dimensions;
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    int num_components;
    try {
	itk_image_get_props (fname, &num_dimensions, 
	    pixelType, componentType, &num_components);
	switch (componentType) {
	case itk::ImageIOBase::UCHAR:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_UCHAR);
	    return load_any_2 (fname, static_cast<unsigned char>(0), otype);
	case itk::ImageIOBase::CHAR:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_CHAR);
	    return load_any_2 (fname, static_cast<char>(0), otype);
	case itk::ImageIOBase::USHORT:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_USHORT);
	    return load_any_2 (fname, static_cast<unsigned short>(0), otype);
	case itk::ImageIOBase::SHORT:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_SHORT);
	    return load_any_2 (fname, static_cast<short>(0), otype);
	case itk::ImageIOBase::UINT:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_ULONG);
	    return load_any_2 (fname, static_cast<unsigned int>(0), otype);
	case itk::ImageIOBase::INT:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_LONG);
	    return load_any_2 (fname, static_cast<int>(0), otype);
	case itk::ImageIOBase::ULONG:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_ULONG);
	    return load_any_2 (fname, static_cast<unsigned long>(0), otype);
	case itk::ImageIOBase::LONG:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_LONG);
	    return load_any_2 (fname, static_cast<long>(0), otype);
	case itk::ImageIOBase::FLOAT:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_FLOAT);
	    return load_any_2 (fname, static_cast<float>(0), otype);
	case itk::ImageIOBase::DOUBLE:
	    set_original_type (original_type, PLM_IMG_TYPE_ITK_DOUBLE);
	    return load_any_2 (fname, static_cast<double>(0), otype);
	case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
	default:
	    fprintf (stderr, 
		"Error: unhandled file type for loading image (%d) %s\n", 
		componentType, fname);
	    exit (-1);
	    break;
	}
    }
    catch (itk::ExceptionObject &excep) {
	std::cerr << "ITK xception loading image: " << fname << std::endl;
	std::cerr << excep << std::endl;
	exit (-1);
    }
}

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

/* Explicit instantiations */
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
