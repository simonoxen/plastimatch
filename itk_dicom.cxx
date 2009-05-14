/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
//#include <iomanip>
#include <iostream>
#include <sstream>
#include "plm_config.h"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itk_dicom.h"
#include "print_and_exit.h"

#include "gdcm/src/gdcmFile.h"
#include "gdcm/src/gdcmUtil.h" 

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
    //nameGenerator->SetUseSeriesDetails (false);
    /* GCS: The following is optional.  Do we need it? */
    // nameGenerator->AddSeriesRestriction("0008|0021" );
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

void
save_image_dicom (ShortImageType::Pointer short_img, char* dir_name)
{
    typedef itk::GDCMImageIO ImageIOType;
    typedef itk::NumericSeriesFileNames NamesGeneratorType;
    const int export_as_ct = 1;

    printf ("Output dir = %s\n", dir_name);

    itksys::SystemTools::MakeDirectory (dir_name);


    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    gdcmIO->SetUIDPrefix ("1.2.826.0.1.3680043.8.274.1.2"); 

    std::string tagkey, value;
    itk::MetaDataDictionary& dict = gdcmIO->GetMetaDataDictionary();
    if (export_as_ct) {
	/* Image Type */
	encapsulate (dict, "0008|0008", "AXIAL");
	/* SOP Class UID */
	encapsulate (dict, "0008|0016", "1.2.840.10008.5.1.4.1.1.2");
	/* Modality */
	encapsulate (dict, "0008|0060", "CT");
    } else { /* export as secondary capture */
	/* Image Type */
	encapsulate (dict, "0008|0008", "DERIVED\\SECONDARY");
	/* Conversion Type */
	encapsulate (dict, "0008|0064", "DV");
    }

    /* Patient name */
    encapsulate (dict, "0010|0010", "PLASTIMATCH^ANONYMOUS");
    /* Patient id */
    encapsulate (dict, "0010|0020", "anon");

    /* Frame of Reference UID */
    encapsulate (dict, "0020|0052", gdcm::Util::CreateUniqueUID (gdcmIO->GetUIDPrefix()));
    /* Position Reference Indicator */
    encapsulate (dict, "0020|1040", "");

    /* Slice thickness */
    value = to_string ((double) (short_img->GetSpacing()[2]));
    printf ("value = %g\n", short_img->GetSpacing()[2]);
    printf ("value = %s\n", value.c_str());
    encapsulate (dict, "0018|0050", value);

    /* 0008,2112 is "Source Image Sequence", defined as "A Sequence that 
	identifies the set of Image SOP Class/Instance pairs of the
	Images that were used to derive this Image. Zero or more Items may be
	included in this Sequence." 
       Ideally, this would be used to refer to the original image before 
        warping.
    */

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

#if defined (commentout)
    // Copy the dictionary from the first slice
    CopyDictionary (*inputDict, *dict);

    // Set the UID's for the study, series, SOP  and frame of reference
    itk::EncapsulateMetaData<std::string>(*dict,"0020|000d", studyUID);
    itk::EncapsulateMetaData<std::string>(*dict,"0020|000e", seriesUID);
    itk::EncapsulateMetaData<std::string>(*dict,"0020|0052", frameOfReferenceUID);

    std::string sopInstanceUID = gdcm::Util::CreateUniqueUID( gdcmIO->GetUIDPrefix());
    itk::EncapsulateMetaData<std::string>(*dict,"0008|0018", sopInstanceUID);
    itk::EncapsulateMetaData<std::string>(*dict,"0002|0003", sopInstanceUID);

    // Change fields that are slice specific
    itksys_ios::ostringstream value;
    value.str("");
    value << f + 1;

    // Image Number
    itk::EncapsulateMetaData<std::string>(*dict,"0020|0013", value.str());

    // Series Description - Append new description to current series
    // description
    std::string oldSeriesDesc;
    itk::ExposeMetaData<std::string>(*inputDict, "0008|103e", oldSeriesDesc);

    value.str("");
    value << oldSeriesDesc
          << ": Resampled with pixel spacing "
          << outputSpacing[0] << ", "
          << outputSpacing[1] << ", "
          << outputSpacing[2];
    // This is an long string and there is a 64 character limit in the
    // standard
    unsigned lengthDesc = value.str().length();
   
    std::string seriesDesc( value.str(), 0,
                            lengthDesc > 64 ? 64
                            : lengthDesc);
    itk::EncapsulateMetaData<std::string>(*dict,"0008|103e", seriesDesc);

    // Series Number
    value.str("");
    value << 1001;
    itk::EncapsulateMetaData<std::string>(*dict,"0020|0011", value.str());

    // Derivation Description - How this image was derived
    value.str("");
    for (unsigned int i = 0; i < argc; i++)
      {
      value << argv[i] << " ";
      }
    value << ": " << ITK_SOURCE_VERSION;

    lengthDesc = value.str().length();
    std::string derivationDesc( value.str(), 0,
                                lengthDesc > 1024 ? 1024
                                : lengthDesc);
    itk::EncapsulateMetaData<std::string>(*dict,"0008|2111", derivationDesc);
   
    // Image Position Patient: This is calculated by computing the
    // physical coordinate of the first pixel in each slice.
    InputImageType::PointType position;
    InputImageType::IndexType index;
    index[0] = 0;
    index[1] = 0;
    index[2] = f;
    resampler->GetOutput()->TransformIndexToPhysicalPoint(index, position);

    value.str("");
    value << position[0] << "\\" << position[1] << "\\" << position[2];
    itk::EncapsulateMetaData<std::string>(*dict,"0020|0032", value.str());      
    // Slice Location: For now, we store the z component of the Image
    // Position Patient.
    value.str("");
    value << position[2];
    itk::EncapsulateMetaData<std::string>(*dict,"0020|1041", value.str());      

    if (changeInSpacing)
      {
      // Slice Thickness: For now, we store the z spacing
      value.str("");
      value << outputSpacing[2];
      itk::EncapsulateMetaData<std::string>(*dict,"0018|0050",
                                            value.str());
      // Spacing Between Slices
      itk::EncapsulateMetaData<std::string>(*dict,"0018|0088",
                                            value.str());
      }
     
    // Save the dictionary
    outputArray.push_back(dict);
#endif

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
