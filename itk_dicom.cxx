/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itk_dicom.h"
#include "logfile.h"
#include "plm_image_patient_position.h"
#include "print_and_exit.h"

/* winbase.h defines GetCurrentTime which conflicts with gdcm function */
#if defined GetCurrentTime
# undef GetCurrentTime
#endif

#include "gdcmFile.h"
#if GDCM_MAJOR_VERSION < 2
#include "gdcmUtil.h"
#else
#include "gdcmUIDGenerator.h"
#endif


/* -----------------------------------------------------------------------
    Definitions
   ----------------------------------------------------------------------- */
typedef itk::ImageSeriesReader < UCharImageType > DicomUCharReaderType;
typedef itk::ImageSeriesReader < ShortImageType > DicomShortReaderType;
typedef itk::ImageSeriesReader < UShortImageType > DicomUShortReaderType;
typedef itk::ImageSeriesReader < Int32ImageType > DicomInt32ReaderType;
typedef itk::ImageSeriesReader < UInt32ImageType > DicomUInt32ReaderType;
typedef itk::ImageSeriesReader < FloatImageType > DicomFloatReaderType;
typedef itk::ImageSeriesWriter < ShortImageType, ShortImage2DType > DicomShortWriterType;

/* -----------------------------------------------------------------------
   Reading Dicom
   ----------------------------------------------------------------------- */
template<class T>
void
load_dicom_dir_rdr(T rdr, const char *dicom_dir)
{
    typedef itk::GDCMImageIO ImageIOType;
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    rdr->SetImageIO( dicomIO );

    /* Read the filenames from the directory */
    typedef itk::GDCMSeriesFileNames NamesGeneratorType;
    NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
    nameGenerator->SetUseSeriesDetails (true);
    //nameGenerator->SetUseSeriesDetails (false);
    /* GCS: The following is optional.  Do we need it? */
    // nameGenerator->AddSeriesRestriction("0008|0021" );
    nameGenerator->SetDirectory (dicom_dir);

    try {
	std::cout << std::endl << "The directory: " << std::endl;
	std::cout << std::endl << dicom_dir << std::endl << std::endl;
	std::cout << "Contains the following DICOM Series: ";
	std::cout << std::endl;

	typedef std::vector< std::string > SeriesIdContainer;
	const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
	SeriesIdContainer::const_iterator seriesItr = seriesUID.begin();
	SeriesIdContainer::const_iterator seriesEnd = seriesUID.end();
	while (seriesItr != seriesEnd) {
	    std::cout << seriesItr->c_str() << std::endl;
	    seriesItr++;
	}
	std::cout << std::endl;

	/* Loop through series and use first one that loads */
	seriesItr = seriesUID.begin();
	bool dicom_load_succeeded = false;
	while (!dicom_load_succeeded && seriesItr != seriesEnd) {
	    std::string seriesIdentifier;
	    seriesIdentifier = seriesItr->c_str();

	    std::cout << "Now reading series: " << std::endl;
	    std::cout << seriesIdentifier << std::endl;

	    /* Read the files */
	    typedef std::vector< std::string >   FileNamesContainer;
	    FileNamesContainer fileNames;
	    fileNames = nameGenerator->GetFileNames( seriesIdentifier );

#if defined (commentout)
	    /* Print out the file names */
	    FileNamesContainer::const_iterator fn_it = fileNames.begin();
	    printf ("File names are:\n");
	    while (fn_it != fileNames.end()) {
		printf ("  %s\n", fn_it->c_str());
		fn_it ++;
	    }
#endif

	    rdr->SetFileNames( fileNames );
	    try {
		rdr->Update();
		dicom_load_succeeded = true;
	    } catch (itk::ExceptionObject &ex) {
		/* do nothing */
		logfile_printf ("Failed to load: %s\n", ex.GetDescription());
	    }
	    seriesItr++;
	}
	if (!dicom_load_succeeded) {
	    print_and_exit ("Error, unable to load dicom series.\n");
	}
    } catch (itk::ExceptionObject &ex) {
	std::cout << ex << std::endl;
	print_and_exit ("Error loading dicom series.\n");
    }
}

UCharImageType::Pointer
load_dicom_uchar (const char *dicom_dir)
{
    DicomUCharReaderType::Pointer fixed_input_rdr
		= DicomUCharReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

ShortImageType::Pointer
load_dicom_short (const char *dicom_dir)
{
    DicomShortReaderType::Pointer fixed_input_rdr
		= DicomShortReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

UShortImageType::Pointer
load_dicom_ushort (const char *dicom_dir)
{
    DicomUShortReaderType::Pointer fixed_input_rdr
		= DicomUShortReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

Int32ImageType::Pointer
load_dicom_int32 (const char *dicom_dir)
{
    DicomInt32ReaderType::Pointer fixed_input_rdr
		= DicomInt32ReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

UInt32ImageType::Pointer
load_dicom_uint32 (const char *dicom_dir)
{
    DicomUInt32ReaderType::Pointer fixed_input_rdr
		= DicomUInt32ReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

FloatImageType::Pointer
load_dicom_float (const char *dicom_dir)
{
    DicomFloatReaderType::Pointer fixed_input_rdr
		= DicomFloatReaderType::New();
    load_dicom_dir_rdr (fixed_input_rdr, dicom_dir);
    fixed_input_rdr->Update();
    return fixed_input_rdr->GetOutput();
}

template <typename T>
static std::string to_string (T t)
{
   std::stringstream ss;
   ss << t;
   return ss.str();
}

static void
encapsulate (itk::MetaDataDictionary& dict, std::string tagkey, std::string value)
{
    itk::EncapsulateMetaData<std::string> (dict, tagkey, value);
}

void
CopyDictionary (itk::MetaDataDictionary &fromDict, itk::MetaDataDictionary &toDict)
{
  typedef itk::MetaDataDictionary DictionaryType;

  DictionaryType::ConstIterator itr = fromDict.Begin();
  DictionaryType::ConstIterator end = fromDict.End();
  typedef itk::MetaDataObject< std::string > MetaDataStringType;

  while( itr != end )
    {
    itk::MetaDataObjectBase::Pointer  entry = itr->second;

    MetaDataStringType::Pointer entryvalue =
      dynamic_cast<MetaDataStringType *>( entry.GetPointer() ) ;
    if( entryvalue )
      {
      std::string tagkey   = itr->first;
      std::string tagvalue = entryvalue->GetMetaDataObjectValue();
      itk::EncapsulateMetaData<std::string>(toDict, tagkey, tagvalue);
      }
    ++itr;
    }
}

std::string 
make_anon_patient_id (void)
{
    int i;
    unsigned char uuid[16];
    std::string patient_id = "PL";

    /* Ugh.  It is a private function. */
    //    bool rc = gdcm::Util::GenerateUUID (uuid);

    srand (time (0));
    for (i = 0; i < 16; i++) {
       int r = (int) (10.0 * rand() / RAND_MAX);
       uuid[i] = '0' + r;
    }
    uuid [15] = '\0';
    patient_id = patient_id + to_string (uuid);
    return patient_id;
}

void
itk_dicom_save (
    ShortImageType::Pointer short_img, 
    const char* dir_name, 
    Plm_image_patient_position patient_pos)
{
    typedef itk::GDCMImageIO ImageIOType;
    typedef itk::NumericSeriesFileNames NamesGeneratorType;
    const int export_as_ct = 1;

    const std::string &current_date = gdcm::Util::GetCurrentDate();
    const std::string &current_time = gdcm::Util::GetCurrentTime();

    printf ("Output dir = %s\n", dir_name);

    itksys::SystemTools::MakeDirectory (dir_name);

    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    gdcmIO->SetUIDPrefix ("1.2.826.0.1.3680043.8.274.1.2"); 

    std::string tagkey, value;
    itk::MetaDataDictionary& dict = gdcmIO->GetMetaDataDictionary();
    if (export_as_ct) {
	/* Image Type */
	//encapsulate (dict, "0008|0008", "AXIAL");
	encapsulate (dict, "0008|0008", "DERIVED\\SECONDARY\\REFORMATTED");
	/* SOP Class UID */
	encapsulate (dict, "0008|0016", "1.2.840.10008.5.1.4.1.1.2");
	/* Modality */
	encapsulate (dict, "0008|0060", "CT");
	/* Conversion Type */
	/* Note: Proton XiO does not like conversion type of SYN */
	encapsulate (dict, "0008|0064", "");
    } else { /* export as secondary capture */
	/* Image Type */
	encapsulate (dict, "0008|0008", "DERIVED\\SECONDARY");
	/* Conversion Type */
	encapsulate (dict, "0008|0064", "DV");
    }

    /* StudyDate, SeriesDate, AcquisitionDate */
    encapsulate (dict, "0008|0021", current_date);
    encapsulate (dict, "0008|0022", current_date);
    encapsulate (dict, "0008|0023", current_date);
    /* StudyTime, SeriesTime, AcquisitionTime */
    encapsulate (dict, "0008|0031", current_time);
    encapsulate (dict, "0008|0032", current_time);
    encapsulate (dict, "0008|0033", current_time);

    /* Patient name */
    encapsulate (dict, "0010|0010", "ANONYMOUS");
    /* Patient id */
    encapsulate (dict, "0010|0020", make_anon_patient_id());
    /* Patient sex */
    encapsulate (dict, "0010|0040", "O");

    /* PatientPosition */
    if ( (patient_pos == PATIENT_POSITION_UNKNOWN) || (patient_pos == PATIENT_POSITION_HFS) )
	encapsulate (dict, "0018|5100", "HFS");
    else if (patient_pos == PATIENT_POSITION_HFP)
	encapsulate (dict, "0018|5100", "HFP");
    else if (patient_pos == PATIENT_POSITION_FFS)
	encapsulate (dict, "0018|5100", "FFS");
    else if (patient_pos == PATIENT_POSITION_FFP)
	encapsulate (dict, "0018|5100", "FFP");

    /* StudyId */
    encapsulate (dict, "0020|0010", "10001");
    /* SeriesNumber */
    encapsulate (dict, "0020|0011", "303");

    /* Frame of Reference UID */
#if GDCM_MAJOR_VERSION < 2
    encapsulate (dict, "0020|0052", gdcm::Util::CreateUniqueUID (gdcmIO->GetUIDPrefix()));
#else
    gdcm::UIDGenerator uid;
    encapsulate (dict, "0020|0052", uid.Generate());    
#endif
    /* Position Reference Indicator */
    encapsulate (dict, "0020|1040", "");


    /* Slice thickness */
    value = to_string ((double) (short_img->GetSpacing()[2]));
    encapsulate (dict, "0018|0050", value);

    /* 0008,2112 is "Source Image Sequence", defined as "A Sequence that 
	identifies the set of Image SOP Class/Instance pairs of the
	Images that were used to derive this Image. Zero or more Items may be
	included in this Sequence." 
       Ideally, this would be used to refer to the original image before 
        warping.
    */

    /* CERR requires slope and offset */
    /* Rescale intercept */
    encapsulate (dict, "0028|1052", "0");
    /* Rescale slope */
    encapsulate (dict, "0028|1053", "1");

    /* Can the series writer set Slice Location "0020,1041"? 
	Yes it can.  The below code is adapted from:
	http://www.nabble.com/Read-DICOM-Series-Write-DICOM-Series-with-a-different-number-of-slices-td17357270.html
    */
    DicomShortWriterType::DictionaryArrayType dict_array;
    for (unsigned int f = 0; f < short_img->GetLargestPossibleRegion().GetSize()[2]; f++) {
	DicomShortWriterType::DictionaryRawPointer slice_dict = new DicomShortWriterType::DictionaryType;
	CopyDictionary (dict, *slice_dict);

	/* Image Number */
	value = to_string ((int) f);
	encapsulate (*slice_dict, "0020|0013", value);

	/* Image Position Patient */
	ShortImageType::PointType position;
	ShortImageType::IndexType index;
	index[0] = 0;
	index[1] = 0;
	index[2] = f;
	short_img->TransformIndexToPhysicalPoint (index, position);
	value = to_string (position[0]) + "\\" + to_string (position[1]) + "\\" + to_string (position[2]);
	encapsulate (*slice_dict, "0020|0032", value);

	/* Slice Location */
	value = to_string ((float) position[2]);
	encapsulate (*slice_dict, "0020|1041", value);      

	dict_array.push_back (slice_dict);
    }

    /* Create file names */
    DicomShortWriterType::Pointer seriesWriter = DicomShortWriterType::New();
    NamesGeneratorType::Pointer namesGenerator = NamesGeneratorType::New();

    ShortImageType::RegionType region = short_img->GetLargestPossibleRegion();
    ShortImageType::IndexType start = region.GetIndex();
    ShortImageType::SizeType  size  = region.GetSize();
    std::string format = dir_name;
    format += "/image%03d.dcm";
    namesGenerator->SetSeriesFormat (format.c_str());
    namesGenerator->SetStartIndex (start[2]);
    namesGenerator->SetEndIndex (start[2] + size[2] - 1);
    namesGenerator->SetIncrementIndex (1);

    seriesWriter->SetInput (short_img);
    seriesWriter->SetImageIO (gdcmIO);
    seriesWriter->SetFileNames (namesGenerator->GetFileNames());
    seriesWriter->SetMetaDataDictionaryArray (&dict_array);

    try {
	seriesWriter->Update();
    }
    catch (itk::ExceptionObject & excp) {
	std::cerr << "Exception thrown while writing the series " << std::endl;
	std::cerr << excp << std::endl;
	exit (-1);
    }
}

/* Explicit instantiations */
template void load_dicom_dir_rdr(DicomShortReaderType::Pointer rdr, const char *dicom_dir);
template void load_dicom_dir_rdr(DicomUShortReaderType::Pointer rdr, const char *dicom_dir);
template void load_dicom_dir_rdr(DicomFloatReaderType::Pointer rdr, const char *dicom_dir);
