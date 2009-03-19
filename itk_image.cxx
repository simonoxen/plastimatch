/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif
#include "plm_config.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"
#include "itk_image.h"
#include "print_and_exit.h"
#include "itk_dicom.h"

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
int
is_directory (char *dir)
{
#if (defined(_WIN32) || defined(WIN32))
    char pwd[_MAX_PATH];
    if (!_getcwd (pwd, _MAX_PATH)) {
        return 0;
    }
    if (_chdir (dir) == -1) {
        return 0;
    }
    _chdir (pwd);
#else /* UNIX */
    DIR *dp;
    if ((dp = opendir (dir)) == NULL) {
        return 0;
    }
    closedir (dp);
#endif
    return 1;
}

int
extension_is (char* fname, char* ext)
{
    return (strlen (fname) > strlen(ext)) 
	&& !strcmp (&fname[strlen(fname)-strlen(ext)], ext);
}

// This function is copied from Slicer3 (itkPluginUtilities.h)
//   so it's available in case Slicer3 is not installed.
// Get the PixelType and ComponentType from fileName
void
itk__GetImageType (std::string fileName,
		    itk::ImageIOBase::IOPixelType &pixelType,
		    itk::ImageIOBase::IOComponentType &componentType)
{
    typedef itk::Image<short, 3> ImageType;
    itk::ImageFileReader<ImageType>::Pointer imageReader =
	itk::ImageFileReader<ImageType>::New();
    imageReader->SetFileName(fileName.c_str());
    imageReader->UpdateOutputInformation();

    pixelType = imageReader->GetImageIO()->GetPixelType();
    componentType = imageReader->GetImageIO()->GetComponentType();
}

template<class RdrT>
void
load_itk_rdr(RdrT reader, char *fn)
{
    reader->SetFileName(fn);
    try {
	reader->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception reading mha file: %s!\n",fn);
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
    
    OrienterType::Pointer orienter = OrienterType::New();
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
	    return PLM_IMG_TYPE_ITK_SHORT;
	} else if (!strcmp(buf, "ElementType = MET_USHORT\n")) {
	    return PLM_IMG_TYPE_ITK_USHORT;
	} else if (!strcmp(buf, "ElementType = MET_UCHAR\n")) {
	    return PLM_IMG_TYPE_ITK_UCHAR;
	} else if (!strcmp(buf, "ElementType = MET_FLOAT\n")) {
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
load_any_2 (char* fname, T, U)
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

template<class U>
typename itk::Image< U, 3 >::Pointer
load_any (char* fname, U otype)
{
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    try {
	itk__GetImageType (fname, pixelType, componentType);
	switch (componentType) {
        case itk::ImageIOBase::UCHAR:
	    return load_any_2 (fname, static_cast<unsigned char>(0), otype);
	case itk::ImageIOBase::CHAR:
	    return load_any_2 (fname, static_cast<char>(0), otype);
	case itk::ImageIOBase::USHORT:
	    return load_any_2 (fname, static_cast<unsigned short>(0), otype);
	case itk::ImageIOBase::SHORT:
	    return load_any_2 (fname, static_cast<short>(0), otype);
	case itk::ImageIOBase::UINT:
	    return load_any_2 (fname, static_cast<unsigned int>(0), otype);
	case itk::ImageIOBase::INT:
	    return load_any_2 (fname, static_cast<int>(0), otype);
	case itk::ImageIOBase::ULONG:
	    return load_any_2 (fname, static_cast<unsigned long>(0), otype);
	case itk::ImageIOBase::LONG:
	    return load_any_2 (fname, static_cast<long>(0), otype);
	case itk::ImageIOBase::FLOAT:
	    return load_any_2 (fname, static_cast<float>(0), otype);
	case itk::ImageIOBase::DOUBLE:
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
	std::cerr << "Exception loading image: " << fname << std::endl;
	std::cerr << excep << std::endl;
	exit (-1);
    }
}

UCharImageType::Pointer
load_uchar (char* fname)
{
    UCharImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_uchar (fname);
    } else {
	img = load_any (fname, static_cast<unsigned char>(0));
    }
    return orient_image (img);
}

ShortImageType::Pointer
load_short (char* fname)
{
    ShortImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_short (fname);
    } else {
	img = load_any (fname, static_cast<short>(0));
    }
    return orient_image (img);
}

UShortImageType::Pointer
load_ushort (char* fname)
{
    UShortImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img load_dicom_ushort (fname);
    } else {
	img load_any (fname, static_cast<unsigned short>(0));
    }
    return orient_image (img);
}

FloatImageType::Pointer
load_float (PlmImageType* original_type, char* fname)
{
    FloatImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	img = load_dicom_float (fname);
    } else {
	img = load_any (fname, static_cast<float>(0));
    }
    return orient_image (img);
}

FloatImageType::Pointer
load_float (char* fname)
{
    FloatImageType::Pointer img;

    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	return load_dicom_float (fname);
    } else {
	return load_any (fname, static_cast<float>(0));
    }
    return orient_image (img);
}

DeformationFieldType::Pointer
load_float_field (char* fname)
{
    typedef itk::ImageFileReader< DeformationFieldType >  FieldReaderType;

    FieldReaderType::Pointer fieldReader = FieldReaderType::New();
    fieldReader->SetFileName (fname);

    try 
    {
	fieldReader->Update();
    }
    catch (itk::ExceptionObject& excp) 
    {
	std::cerr << "Exception thrown " << std::endl;
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
save_image (T image, char* fname)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageFileWriter< ImageType >  WriterType;

    printf ("Trying to write image to %s\n", fname);

    typename WriterType::Pointer writer = WriterType::New();
    writer->SetInput (image);
    writer->SetFileName (fname);
    try {
	writer->Update();
    }
    catch (itk::ExceptionObject& excp) {
	std::cerr << "Exception thrown " << std::endl;
	std::cerr << excp << std::endl;
    }
}

template<class T> 
void
save_short (T image, char* fname)
{
    ShortImageType::Pointer short_img = cast_short(image);
    save_image (short_img, fname);
}

template<class T> 
void
save_short_dicom (T image, char* dir_name)
{
    ShortImageType::Pointer short_img = cast_short(image);
    save_image_dicom (short_img, dir_name);
}

template<class T> 
void
save_float (T image, char* fname)
{
    FloatImageType::Pointer float_img = cast_float(image);
    save_image (float_img, fname);
}

/* -----------------------------------------------------------------------
   Casting image types
   ----------------------------------------------------------------------- */
template<class T> 
ShortImageType::Pointer
cast_short (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, ShortImageType > CastFilterType;

    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput(image);
    caster->Update();
    return caster->GetOutput();
}

template<class T> 
FloatImageType::Pointer
cast_float (T image)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::CastImageFilter <
	ImageType, FloatImageType > CastFilterType;

    typename CastFilterType::Pointer caster = CastFilterType::New();
    caster->SetInput(image);
    caster->Update();
    return caster->GetOutput();
}

/* Explicit instantiations */
template plastimatch1_EXPORT void save_image(UCharImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(ShortImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(UShortImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(DeformationFieldType::Pointer, char*);
template void save_short (FloatImageType::Pointer, char*);
template void save_short_dicom (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void save_float (FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template plastimatch1_EXPORT void get_image_header (int dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
