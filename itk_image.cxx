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
#include "itk_dicom.h"
#include "itk_image_cast.h"
#include "file_util.h"
#include "print_and_exit.h"
#include "logfile.h"

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
// This function is copied from Slicer3 (itkPluginUtilities.h)
//   so it's available in case Slicer3 is not installed.
// Get the PixelType and ComponentType from fileName
void
itk__GetImageType (std::string fileName,
		    itk::ImageIOBase::IOPixelType &pixel_type,
		    itk::ImageIOBase::IOComponentType &component_type)
{
    pixel_type = itk::ImageIOBase::UNKNOWNPIXELTYPE;
    component_type = itk::ImageIOBase::UNKNOWNCOMPONENTTYPE;
    typedef itk::Image<short, 3> ImageType;
    itk::ImageFileReader<ImageType>::Pointer imageReader =
	itk::ImageFileReader<ImageType>::New();
    imageReader->SetFileName(fileName.c_str());
    try {
	imageReader->UpdateOutputInformation();
	pixel_type = imageReader->GetImageIO()->GetPixelType();
	component_type = imageReader->GetImageIO()->GetComponentType();
    } catch (itk::ExceptionObject &ex) {
	ex;    /* Suppress compiler warning */
    }
}

template<class RdrT>
void
load_itk_rdr (RdrT reader, const char *fn)
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
int
get_mha_type (char* mha_fname)
{
    char buf[1024];
    FILE* fp = fopen (mha_fname, "r");
    if (!fp) {
	printf ("Could not open mha file for read\n");
	exit (-1);
    }
    while (fgets(buf,1024,fp)) {
	if (!strcmp(buf, "ElementType = MET_SHORT\n")) {
	    fclose (fp);
	    return PLM_IMG_TYPE_ITK_SHORT;
	} else if (!strcmp(buf, "ElementType = MET_USHORT\n")) {
	    fclose (fp);
	    return PLM_IMG_TYPE_ITK_USHORT;
	} else if (!strcmp(buf, "ElementType = MET_UCHAR\n")) {
	    fclose (fp);
	    return PLM_IMG_TYPE_ITK_UCHAR;
	} else if (!strcmp(buf, "ElementType = MET_FLOAT\n")) {
	    fclose (fp);
	    return PLM_IMG_TYPE_ITK_FLOAT;
	} else if (!strncmp(buf,"ElementType",sizeof("ElementType"))) {
	    printf ("No ElementType in mha file\n");
	    exit (-1);
	}
    }
    printf ("No ElementType in mha file\n");
    exit (-1);
    return 0;  /* Get rid of warning */
}

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
    load_itk_rdr (rdr, fname);
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
set_original_type (PlmImageType *original_type,
		   PlmImageType t)
{
    if (original_type) {
	*original_type = t;
    }
}

template<class U>
typename itk::Image< U, 3 >::Pointer
load_any (const char* fname,
	  PlmImageType* original_type, 
	  U otype)
{
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    try {
	itk__GetImageType (fname, pixelType, componentType);
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
		     "Error: unhandled file type for loading image %s\n",
		     fname);
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
load_uchar (const char* fname, PlmImageType* original_type)
{
    UCharImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uchar (fname);
    } else {
	img = load_any (fname, original_type, static_cast<unsigned char>(0));
    }
    return orient_image (img);
}

ShortImageType::Pointer
load_short (const char* fname, PlmImageType* original_type)
{
    ShortImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_short (fname);
    } else {
	img = load_any (fname, original_type, static_cast<short>(0));
    }
    return orient_image (img);
}

UShortImageType::Pointer
load_ushort (const char* fname, PlmImageType* original_type)
{
    UShortImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_ushort (fname);
    } else {
	img = load_any (fname, original_type, static_cast<unsigned short>(0));
    }
    return orient_image (img);
}

UInt32ImageType::Pointer
load_uint32 (const char* fname, PlmImageType* original_type)
{
    UInt32ImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uint32 (fname);
    } else {
	img = load_any (fname, original_type, static_cast<uint32_t>(0));
    }
    return orient_image (img);
}

FloatImageType::Pointer
load_float (const char* fname, PlmImageType* original_type)
{
    FloatImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_float (fname);
    } else {
	img = load_any (fname, original_type, static_cast<float>(0));
    }
    return orient_image (img);
}

DeformationFieldType::Pointer
load_float_field (const char* fname)
{
    typedef itk::ImageFileReader< DeformationFieldType >  FieldReaderType;

    FieldReaderType::Pointer fieldReader = FieldReaderType::New();
    fieldReader->SetFileName (fname);

    try {
	fieldReader->Update();
    }
    catch (itk::ExceptionObject& excp) {
	std::cerr << "ITK exception reading vf file." << std::endl;
	std::cerr << excp << std::endl;
	return 0;
    }
    DeformationFieldType::Pointer deform_field = fieldReader->GetOutput();
    return deform_field;
}

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
    try {
	writer->Update();
    }
    catch (itk::ExceptionObject& excp) {
	std::cerr << "ITK exception writing image file." << std::endl;
	std::cerr << excp << std::endl;
    }
}

template<class T> 
void
itk_image_save_uchar (T image, char* fname)
{
    UCharImageType::Pointer uchar_img = cast_uchar (image);
    itk_image_save (uchar_img, fname);
}

template<class T> 
void
itk_image_save_short (T image, char* fname)
{
    ShortImageType::Pointer short_img = cast_short (image);
    itk_image_save (short_img, fname);
}

template<class T> 
void
itk_image_save_ushort (T image, char* fname)
{
    UShortImageType::Pointer ushort_img = cast_ushort (image);
    itk_image_save (ushort_img, fname);
}

template<class T> 
void
itk_image_save_short_dicom (T image, char* dir_name)
{
    ShortImageType::Pointer short_img = cast_short (image);
    itk_dicom_save (short_img, dir_name);
}

template<class T> 
void
itk_image_save_uint32 (T image, char* fname)
{
    UInt32ImageType::Pointer uint32_img = cast_uint32 (image);
    itk_image_save (uint32_img, fname);
}

template<class T> 
void
itk_image_save_float (T image, char* fname)
{
    FloatImageType::Pointer float_img = cast_float (image);
    itk_image_save (float_img, fname);
}

/* Explicit instantiations */
template plastimatch1_EXPORT void itk_image_save(UCharImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(ShortImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(UShortImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(UInt32ImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(FloatImageType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save(DeformationFieldType::Pointer, const char*);
template plastimatch1_EXPORT void itk_image_save_uchar (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_ushort (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_short (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_uint32 (FloatImageType::Pointer, char*);

template plastimatch1_EXPORT void itk_image_save_short_dicom (UCharImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_short_dicom (ShortImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_short_dicom (UShortImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_short_dicom (UInt32ImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_short_dicom (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void itk_image_save_float (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
