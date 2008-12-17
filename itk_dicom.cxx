/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itk_dicom.h"
#include "print_and_exit.h"

/* -----------------------------------------------------------------------
    Definitions
   ----------------------------------------------------------------------- */
typedef itk::ImageSeriesReader < UCharImageType > DicomUCharReaderType;
typedef itk::ImageSeriesReader < ShortImageType > DicomShortReaderType;
typedef itk::ImageSeriesReader < UShortImageType > DicomUShortReaderType;
typedef itk::ImageSeriesReader < FloatImageType > DicomFloatReaderType;
typedef itk::ImageSeriesWriter < ShortImageType, ShortImage2DType > DicomShortWriterType;

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

static void
encapsulate (itk::MetaDataDictionary& dict, std::string tagkey, std::string value)
{
    itk::EncapsulateMetaData<std::string> (dict, tagkey, value);
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

    encapsulate (dict, "0008|0008", "DERIVED\\SECONDARY");

//    tagkey = "0008|0008"; // Image Type
//    value = "DERIVED\\SECONDARY";
//    itk::EncapsulateMetaData<std::string>(dict, tagkey, value);
    tagkey = "0008|0016"; // SOPClassUID
    value = "1.2.840.10008.5.1.4.1.1.2";
    itk::EncapsulateMetaData<std::string>(dict, tagkey, value);
    tagkey = "0008|0060"; // Modality
    value = "CT";
    itk::EncapsulateMetaData<std::string>(dict, tagkey, value );

//    tagkey = "0008|0064"; // Conversion Type
//    value = "DV";
//    itk::EncapsulateMetaData<std::string>(dict, tagkey, value);

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

/* Explicit instantiations */
template void load_dicom_dir_rdr(DicomShortReaderType::Pointer rdr, char *dicom_dir);
template void load_dicom_dir_rdr(DicomUShortReaderType::Pointer rdr, char *dicom_dir);
template void load_dicom_dir_rdr(DicomFloatReaderType::Pointer rdr, char *dicom_dir);
