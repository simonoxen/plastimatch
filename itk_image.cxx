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
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkCastImageFilter.h"
#include "itk_image.h"
#include "print_and_exit.h"

#if (defined(_WIN32) || defined(WIN32))
#define snprintf _snprintf
#define mkdir(a,b) _mkdir(a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

/* =======================================================================*
    Definitions
 * =======================================================================*/
typedef itk::ImageSeriesReader < UCharImageType > DicomUCharReaderType;
typedef itk::ImageSeriesReader < ShortImageType > DicomShortReaderType;
typedef itk::ImageSeriesReader < UShortImageType > DicomUShortReaderType;
typedef itk::ImageSeriesReader < FloatImageType > DicomFloatReaderType;
typedef itk::ImageSeriesWriter < ShortImageType, ShortImage2DType > DicomShortWriterType;
typedef itk::ImageFileReader < UCharImageType > MhaUCharReaderType;
typedef itk::ImageFileReader < ShortImageType > MhaShortReaderType;
typedef itk::ImageFileReader < UShortImageType > MhaUShortReaderType;
typedef itk::ImageFileReader < FloatImageType > MhaFloatReaderType;


/* =======================================================================*
    Functions
 * =======================================================================*/
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
   Reading Dicom
   ----------------------------------------------------------------------- */
template<class T>
void
load_dicom_dir_rdr(T rdr, char *dicom_dir)
{
    typedef itk::GDCMImageIO ImageIOType;
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    rdr->SetImageIO( dicomIO );

    /* Read the filenames from the directory */
    typedef itk::GDCMSeriesFileNames NamesGeneratorType;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetUseSeriesDetails (true);
    /* GCS: The following is optional.  Do we need it? */
    //    nameGenerator->AddSeriesRestriction("0008|0021" );
    nameGenerator->SetDirectory (dicom_dir);

    try {
	std::cout << std::endl << "The directory: " << std::endl;
	std::cout << std::endl << dicom_dir << std::endl << std::endl;
	std::cout << "Contains the following DICOM Series: ";
	std::cout << std::endl << std::endl;

	typedef std::vector< std::string > SeriesIdContainer;
	const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
	SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
	SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
	while (seriesItr != seriesEnd) {
	    std::cout << seriesItr->c_str() << std::endl;
	    seriesItr++;
	}

	/* Assumes series is first one found */
	std::string seriesIdentifier;
	seriesIdentifier = seriesUID.begin()->c_str();

	std::cout << std::endl << std::endl;
	std::cout << "Now reading series: " << std::endl << std::endl;
	std::cout << seriesIdentifier << std::endl;
	std::cout << std::endl << std::endl;

	/* Read the files */
	typedef std::vector< std::string >   FileNamesContainer;
	FileNamesContainer fileNames;
	fileNames = nameGenerator->GetFileNames( seriesIdentifier );
	rdr->SetFileNames( fileNames );
	try {
	    rdr->Update();
	} catch (itk::ExceptionObject &ex) {
	    std::cout << ex << std::endl;
	    print_and_exit ("Error loading dicom series.\n");
	}	
    } catch (itk::ExceptionObject &ex) {
	std::cout << ex << std::endl;
	print_and_exit ("Error loading dicom series.\n");
    }
}

UCharImageType::Pointer
load_dicom_uchar (char *dicom_dir)
{
    DicomUCharReaderType::Pointer fixed_input_rdr
		= DicomUCharReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

ShortImageType::Pointer
load_dicom_short (char *dicom_dir)
{
    DicomShortReaderType::Pointer fixed_input_rdr
		= DicomShortReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

UShortImageType::Pointer
load_dicom_ushort (char *dicom_dir)
{
    DicomUShortReaderType::Pointer fixed_input_rdr
		= DicomUShortReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

FloatImageType::Pointer
load_dicom_float (char *dicom_dir)
{
    DicomFloatReaderType::Pointer fixed_input_rdr
		= DicomFloatReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
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
	    return TYPE_SHORT;
	} else if (!strcmp(buf, "ElementType = MET_USHORT\n")) {
	    return TYPE_USHORT;
	} else if (!strcmp(buf, "ElementType = MET_UCHAR\n")) {
	    return TYPE_UCHAR;
	} else if (!strcmp(buf, "ElementType = MET_FLOAT\n")) {
	    return TYPE_FLOAT;
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
    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	return load_dicom_uchar (fname);
    } else {
	return load_any (fname, static_cast<unsigned char>(0));
    }
}

ShortImageType::Pointer
load_short (char* fname)
{
    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	return load_dicom_short (fname);
    } else {
	return load_any (fname, static_cast<short>(0));
    }
}

UShortImageType::Pointer
load_ushort (char* fname)
{
    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	return load_dicom_ushort (fname);
    } else {
	return load_any (fname, static_cast<unsigned short>(0));
    }
}

FloatImageType::Pointer
load_float (char* fname)
{
    /* If it is directory, then must be dicom */
    if (is_directory(fname)) {
	return load_dicom_float (fname);
    } else {
	return load_any (fname, static_cast<float>(0));
    }
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

void
save_image_dicom (ShortImageType::Pointer short_img, char* dir_name)
{
    typedef itk::GDCMImageIO                        ImageIOType;
    //typedef itk::GDCMSeriesFileNames                NamesGeneratorType;
    typedef itk::NumericSeriesFileNames             NamesGeneratorType;

    printf ("Output dir = %s\n", dir_name);

    itksys::SystemTools::MakeDirectory (dir_name);

    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    DicomShortWriterType::Pointer seriesWriter = DicomShortWriterType::New();
    NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();

    itk::MetaDataDictionary & dict = gdcmIO->GetMetaDataDictionary();
    std::string tagkey, value;
    tagkey = "0008|0060"; // Modality
    value = "CT";
    itk::EncapsulateMetaData<std::string>(dict, tagkey, value );
    tagkey = "0008|0008"; // Image Type
    value = "DERIVED\\SECONDARY";
    itk::EncapsulateMetaData<std::string>(dict, tagkey, value);
    tagkey = "0008|0064"; // Conversion Type
    value = "DV";
    itk::EncapsulateMetaData<std::string>(dict, tagkey, value);

    /* Create file names */
    ShortImageType::RegionType region = short_img->GetLargestPossibleRegion();
    ShortImageType::IndexType start = region.GetIndex();
    ShortImageType::SizeType  size  = region.GetSize();
    std::string format = dir_name;
    format += "/image%03d.dcm";
    namesGenerator->SetSeriesFormat( format.c_str() );
    namesGenerator->SetStartIndex( start[2] );
    namesGenerator->SetEndIndex( start[2] + size[2] - 1 );
    namesGenerator->SetIncrementIndex( 1 );

    seriesWriter->SetInput (short_img);
    seriesWriter->SetImageIO (gdcmIO);
    seriesWriter->SetFileNames (namesGenerator->GetFileNames());

#if defined (commentout)
    seriesWriter->SetMetaDataDictionaryArray (
                        reader->GetMetaDataDictionaryArray() );
#endif

    try {
	seriesWriter->Update();
    } catch (itk::ExceptionObject & excp) {
	std::cerr << "Exception thrown while writing the series " << std::endl;
	std::cerr << excp << std::endl;
	exit (-1);
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
template void load_itk_rdr (MhaUCharReaderType::Pointer reader, char *fn);
template void load_dicom_dir_rdr(DicomShortReaderType::Pointer rdr, char *dicom_dir);
template void load_dicom_dir_rdr(DicomUShortReaderType::Pointer rdr, char *dicom_dir);
template void load_dicom_dir_rdr(DicomFloatReaderType::Pointer rdr, char *dicom_dir);
template plastimatch1_EXPORT void save_image(UCharImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(ShortImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(UShortImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(FloatImageType::Pointer, char*);
template plastimatch1_EXPORT void save_image(DeformationFieldType::Pointer, char*);
template void save_short (FloatImageType::Pointer, char*);
template void save_short_dicom (FloatImageType::Pointer, char*);
template void save_float (FloatImageType::Pointer, char*);
template void get_image_header (int dim[3], float offset[3], float spacing[3], UCharImageType::Pointer image);
template void get_image_header (int dim[3], float offset[3], float spacing[3], ShortImageType::Pointer image);
template void get_image_header (int dim[3], float offset[3], float spacing[3], UShortImageType::Pointer image);
template void get_image_header (int dim[3], float offset[3], float spacing[3], FloatImageType::Pointer image);
