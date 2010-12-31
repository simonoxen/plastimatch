/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkGDCMImageIO.h"
#include "itkImageIOBase.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"
#include "gdcmGlobal.h"

void usage () {
    std::cout << "Usage: dicom_info dicom_file_name" << std::endl;
}

int main (int argc, char* argv[])
{
  if (argc == 1) {
	  usage();
	  return 0;
	  }

  typedef signed short PixelType;
  const unsigned int Dimension = 2;
  typedef itk::Image<PixelType, Dimension> ImageType;

  typedef itk::ImageFileReader<ImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  
  typedef itk::GDCMImageIO ImageIOType;
  ImageIOType::Pointer dicomIO = ImageIOType::New();
  dicomIO->SetMaxSizeLoadEntry(0xffff);

  reader->SetFileName(argv[1]);
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
      if (found)
        std::cout << labelId << " = " << tagvalue.c_str() << std::endl;
      }
    ++itr;
    } 
  
  return EXIT_SUCCESS;
}  
