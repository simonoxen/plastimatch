/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */

#include "plm_config.h"

#include <string.h>

#include "itkImageFileReader.h"
#include "itkGDCMImageIO.h"
#include "itkImageIOBase.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "getopt.h"
#include "gdcmGlobal.h"

void usage () {
    std::cout << "Usage: dicom_info --input=dicom_file_name [--short]" << std::endl;
}

int main (int argc, char* argv[])
{
	
    if (argc == 1) {
	usage();
	return 0;
    }
	
    char * fn = 0;
    bool short_parms = false;

    static const char *optString = "is?";

    static const struct option longOpts[] = {
	{ "input", required_argument, NULL, 'i' },
	{ "short", no_argument, NULL, 's' }
    };

    int longIndex = 0;
    int opt = getopt_long(argc, argv, optString, longOpts, &longIndex);

    while( opt != -1 ) {
        switch( opt ) {
	case 'i':
	    if (!strcmp(optarg,"") || !strcmp(optarg,"=")) {
		usage();
		return 0;
	    }	
	
	    else
		fn=optarg;
	    break;
                
	case 's':
	    short_parms = true;
	    break;
                
	case '?':
	    usage();
	    return 0;
        }
        
        opt = getopt_long( argc, argv, optString, longOpts, &longIndex );
    }


    typedef signed short PixelType;
    const unsigned int Dimension = 2;
    typedef itk::Image<PixelType, Dimension> ImageType;

    typedef itk::ImageFileReader<ImageType> ReaderType;
    ReaderType::Pointer reader = ReaderType::New();
  
    typedef itk::GDCMImageIO ImageIOType;
    ImageIOType::Pointer dicomIO = ImageIOType::New();
    dicomIO->SetMaxSizeLoadEntry(0xffff);

    reader->SetFileName(fn);
    reader->SetImageIO(dicomIO);
  
    try
    {
	reader->Update();
    }
    catch (itk::ExceptionObject &ex)
    {
	std::cout << ex << std::endl;
	return EXIT_FAILURE;
    }

    typedef itk::MetaDataDictionary DictionaryType;
    const  DictionaryType & dictionary = dicomIO->GetMetaDataDictionary();
    typedef itk::MetaDataObject< std::string > MetaDataStringType;
 
    DictionaryType::ConstIterator itr = dictionary.Begin();
    DictionaryType::ConstIterator end = dictionary.End();
 
    while (itr != end)
    {
	itk::MetaDataObjectBase::Pointer entry = itr->second;
	MetaDataStringType::Pointer entryvalue = dynamic_cast <MetaDataStringType *> (entry.GetPointer());
   
	if (entryvalue)
	{
	    std::string tagkey = itr->first;
	    std::string labelId;
	    bool found =  itk::GDCMImageIO::GetLabelFromTag (tagkey, labelId);
	    std::string tagvalue = entryvalue->GetMetaDataObjectValue();

	    if (found) {

		if (short_parms) {

		    if (tagkey == "0010|0010" || // Patient's Name
			tagkey == "0010|0040" || // Patient's Sex
			tagkey == "0010|0030" || // Patient's Birth Date
			tagkey == "0010|1010" || // Patient's Age
			tagkey == "0008|0020" || // Study Date
			tagkey == "0008|0080" || // Institution Name
			tagkey == "0008|0090" || // Referring Physician's Name
			tagkey == "0008|0060" || // Modality
			tagkey == "0028|0010" || // Rows
			tagkey == "0028|0011" || // Columns
			tagkey == "0028|0030" || // Pixel Spacing
			tagkey == "0018|0050")   // Slice Thickness

			std::cout << labelId << " = " << tagvalue.c_str() << std::endl;
		  
		}
		else 
		    std::cout << labelId << " = " << tagvalue.c_str() << std::endl;
	    }
	}
	++itr;
    } 
  
    return EXIT_SUCCESS;
}  
