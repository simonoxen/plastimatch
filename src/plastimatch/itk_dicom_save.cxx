/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <vector>
#include "itksys/SystemTools.hxx"
#include "itkGDCMImageIO.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkNumericSeriesFileNames.h"
#include "itkImageSeriesWriter.h"
#include "dcm_util.h"
#include "gdcm1_util.h"
#include "gdcm2_util.h"
#include "img_metadata.h"
#include "itk_dicom_save.h"
#include "itk_image.h"
#include "logfile.h"
#include "make_string.h"
#include "print_and_exit.h"
#include "referenced_dicom_dir.h"

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

typedef itk::ImageSeriesWriter < 
    ShortImageType, ShortImage2DType > DicomShortWriterType;
typedef itk::GDCMImageIO ImageIOType;

static void
encapsulate (
    itk::MetaDataDictionary& dict, 
    std::string tagkey, 
    std::string value
)
{
    itk::EncapsulateMetaData<std::string> (dict, tagkey, value);
}

static const std::string
itk_make_uid (ImageIOType::Pointer& gdcmIO)
{
#if GDCM_MAJOR_VERSION < 2
    return gdcm::Util::CreateUniqueUID (gdcmIO->GetUIDPrefix());
#else
    static gdcm::UIDGenerator uid;
    return uid.Generate();
#endif
}

static void
copy_dictionary (
    itk::MetaDataDictionary &fromDict, 
    itk::MetaDataDictionary &toDict
)
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
itk_dicom_save (
    ShortImageType::Pointer short_img,    /* Input: image to write */
    const char *dir_name,                 /* Input: name of output dir */
    Referenced_dicom_dir *rdd,            /* Output: gets filled in */
    const Img_metadata *meta              /* Input: output files get these */
)
{
    typedef itk::NumericSeriesFileNames NamesGeneratorType;
    const int export_as_ct = 1;
    /* DICOM date string looks like this: 20110601
       DICOM time string looks like this: 203842 or 203842.805219
    */
    std::string current_date, current_time;
#if GDCM_VERSION_1
    gdcm1_get_date_time (&current_date, &current_time);
#elif GDCM_VERSION_2
    gdcm2_get_date_time (&current_date, &current_time);
#else
    current_date = "20000101";
    current_time = "000000";
#endif

    itksys::SystemTools::MakeDirectory (dir_name);

    ImageIOType::Pointer gdcmIO = ImageIOType::New();
    gdcmIO->SetUIDPrefix ("1.2.826.0.1.3680043.8.274.1.2"); 

    /* There is apparently no way to get the ITK-generated UIDs 
       out of ITK without re-reading the files.  So we tell ITK not 
       to make them.  Instead generate them here, and add them to 
       the dictionary. */
    gdcmIO->SetKeepOriginalUID (true);

    /* Set up a few things in referenced_dicom_dir */
    if (rdd) {
	rdd->m_loaded = true;
	rdd->m_pih.set_from_itk_image (short_img);
	rdd->m_ct_slice_uids.clear();
    }

    std::string tagkey, value;
    itk::MetaDataDictionary& dict = gdcmIO->GetMetaDataDictionary();
    if (export_as_ct) {
	/* Image Type */
	encapsulate (dict, "0008|0008", "DERIVED\\SECONDARY\\REFORMATTED");
	/* SOP Class UID */
	encapsulate (dict, "0008|0016", "1.2.840.10008.5.1.4.1.1.2");
	/* Modality */
	encapsulate (dict, "0008|0060", "CT");
	/* Conversion Type */
	/* Note: Proton XiO does not like conversion type of SYN */
	encapsulate (dict, "0008|0064", "");
    } else {
	/* Export as secondary capture */
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

    /* Manufacturer */
    encapsulate (dict, "0008|0070", "Plastimatch");
    /* InstitutionName */
    encapsulate (dict, "0008|0080", "");
    /* ManufacturersModelName */
    encapsulate (dict, "0008|1090", "Plastimatch");

    if (meta) {
	/* Patient name */
	encapsulate (dict, "0010|0010", meta->get_metadata (0x0010, 0x0010));
	/* Patient id */
	encapsulate (dict, "0010|0020", meta->get_metadata (0x0010, 0x0020));
	/* Patient sex */
	encapsulate (dict, "0010|0040", meta->get_metadata (0x0010, 0x0040));
    } else {
	/* Patient name */
	encapsulate (dict, "0010|0010", "ANONYMOUS");
	/* Patient id */
	encapsulate (dict, "0010|0020", dcm_anon_patient_id());
	/* Patient sex */
	encapsulate (dict, "0010|0040", "O");
    }

    /* Slice thickness */
    value = make_string ((double) (short_img->GetSpacing()[2]));
    encapsulate (dict, "0018|0050", value);

    /* Patient position */
    value = meta->get_metadata (0x0018, 0x5100);
    if (value == "HFS" || value == "FFS" 
	|| value == "HFP" || value == "FFP" 
	|| value == "HFDL" || value == "HFDR" 
	|| value == "FFDL" || value == "FFDR")
    {
	encapsulate (dict, "0018|5100", value);
    } else {
	encapsulate (dict, "0018|5100", "HFS");
    }

    /* StudyInstanceUID */
    value = itk_make_uid(gdcmIO);
    encapsulate (dict, "0020|000d", value);
    if (rdd) {
	rdd->m_ct_study_uid = value.c_str();
    }
    /* SeriesInstanceUID */
    value = itk_make_uid(gdcmIO);
    encapsulate (dict, "0020|000e", value);
    if (rdd) {
	rdd->m_ct_series_uid = value.c_str();
    }
    /* StudyId */
    value = "10001";
    encapsulate (dict, "0020|0010", value);
    if (rdd) {
	rdd->m_study_id = value.c_str();
    }
    /* SeriesNumber */
    encapsulate (dict, "0020|0011", "303");
    /* Frame of Reference UID */
    value = itk_make_uid(gdcmIO);
    encapsulate (dict, "0020|0052", value);
    if (rdd) {
	rdd->m_ct_fref_uid = value.c_str();
    }
    /* Position Reference Indicator */
    encapsulate (dict, "0020|1040", "");

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
    for (unsigned int f = 0; 
	 f < short_img->GetLargestPossibleRegion().GetSize()[2]; 
	 f++) 
    {
	DicomShortWriterType::DictionaryRawPointer slice_dict 
	    = new DicomShortWriterType::DictionaryType;
	copy_dictionary (dict, *slice_dict);

	/* SOPInstanceUID */
	value = itk_make_uid(gdcmIO);
	encapsulate (*slice_dict, "0008|0018", value);
	if (rdd) {
	    rdd->m_ct_slice_uids.push_back(value.c_str());
	}
	
	/* Image Number */
	value = make_string ((int) f);
	encapsulate (*slice_dict, "0020|0013", value);

	/* Image Position Patient */
	ShortImageType::PointType position;
	ShortImageType::IndexType index;
	index[0] = 0;
	index[1] = 0;
	index[2] = f;
	short_img->TransformIndexToPhysicalPoint (index, position);
	value = make_string (position[0]) 
	    + "\\" + make_string (position[1]) 
	    + "\\" + make_string (position[2]);
	encapsulate (*slice_dict, "0020|0032", value);

	/* Slice Location */
	value = make_string ((float) position[2]);
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
