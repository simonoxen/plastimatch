/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This program is used to convert from dicom to mha */
#include <time.h>
#include "config.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkDICOMImageIO2.h"
#include "itkImageSeriesReader.h"
#include "itkDICOMSeriesFileNames.h"
#include "resample_mha.h"

/* We only deal with these kinds of images... */
const unsigned int Dimension = 3;
typedef short InputPixelType;
typedef float InternalPixelType;

typedef itk::Image < InputPixelType, Dimension > InputImageType;
typedef itk::Image < InternalPixelType, Dimension > InternalImageType;
typedef itk::Image < signed short, Dimension > SignedImageType;
typedef itk::Image < unsigned short, Dimension > UnsignedImageType;
typedef itk::Image < float, Dimension > FloatImageType;

typedef itk::ImageSeriesReader < InputImageType > DicomReaderType;
typedef itk::ImageFileReader < InputImageType > MhaReaderType;
typedef itk::ImageSeriesReader < SignedImageType > SignedDicomReaderType;
typedef itk::ImageSeriesReader < UnsignedImageType > UnsignedDicomReaderType;
typedef itk::ImageSeriesReader < FloatImageType > FloatDicomReaderType;


typedef itk::CastImageFilter< 
                    InputImageType, InternalImageType > FixedCastFilterType;
typedef itk::CastImageFilter< 
                    InputImageType, InternalImageType > MovingCastFilterType;

typedef itk::CastImageFilter<InputImageType, UnsignedImageType> UnsignedCastFilterType;
typedef itk::CastImageFilter<InputImageType, SignedImageType> SignedCastFilterType;

MhaReaderType::Pointer
load_mha_rdr(char *fn)
{
    MhaReaderType::Pointer reader = MhaReaderType::New();
    reader->SetFileName(fn);
    try 
    {
	    printf ("Running update\n");
	    reader->Update();
	    printf ("Done with update\n");
    }
    catch(itk::ExceptionObject & ex) {
	    printf ("Exception reading mha file: %s!\n",fn);
	    std::cout << ex << std::endl;
	    exit(1);
    }
    return reader;
}

InputImageType::Pointer
load_mha(char *fn)
{
    return load_mha_rdr(fn)->GetOutput();
}

template<class T>
void
load_dicom_dir_rdr(T rdr, char *dicom_dir)
{
    itk::DICOMImageIO2::Pointer dicomIO = itk::DICOMImageIO2::New();

    // Get the DICOM filenames from the directory
    itk::DICOMSeriesFileNames::Pointer nameGenerator =
	    itk::DICOMSeriesFileNames::New();
    nameGenerator->SetDirectory(dicom_dir);

    typedef std::vector < std::string > seriesIdContainer;
    const seriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
    seriesIdContainer::const_iterator seriesItr = seriesUID.begin();
    seriesIdContainer::const_iterator seriesEnd = seriesUID.end();
    std::cout << std::endl << "The directory: " << std::endl;
    std::cout << std::endl << dicom_dir << std::endl << std::endl;
    std::cout << "Contains the following DICOM Series: ";
    std::cout << std::endl << std::endl;

    while (seriesItr != seriesEnd) {
	std::cout << seriesItr->c_str() << std::endl;
	seriesItr++;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Now reading series: " << std::endl << std::endl;

    typedef std::vector < std::string > fileNamesContainer;
    fileNamesContainer fileNames;

    std::cout << seriesUID.begin()->c_str() << std::endl;
    fileNames = nameGenerator->GetFileNames();

    rdr->SetFileNames(fileNames);
    rdr->SetImageIO(dicomIO);

    try {
	rdr->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception reading dicom directory: %s!\n",dicom_dir);
	std::cout << ex << std::endl;
	exit(1);
    }
}

void
shift_pet_values (FloatImageType::Pointer image)
{
    printf ("Shifting values for pet...\n");
    typedef itk::ImageRegionIterator< FloatImageType > IteratorType;
    InputImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	float f = it.Get();
	f = f - 100;
	if (f < 0) {
	    f = f * 10;
	} else if (f > 0x4000) {
	    f = 0x4000 + (f - 0x4000) / 2;
	}
	if (f > 0x07FFF) {
	    f = 0x07FFF;
	}
	it.Set(f);
    }
}

void
fix_invalid_pixels_with_shift(SignedImageType::Pointer image)
{
    typedef itk::ImageRegionIterator< InputImageType > IteratorType;
    InputImageType::RegionType region = image->GetLargestPossibleRegion();
    IteratorType it (image, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	InputPixelType c = it.Get();
	if (c < -1000) {
	    c = -1000;
	}
	it.Set (c + 1000);
    }
}

void
fix_invalid_pixels(SignedImageType::Pointer image)
{
  typedef itk::ImageRegionIterator< InputImageType > IteratorType;
  InputImageType::RegionType region = image->GetLargestPossibleRegion();
  IteratorType it (image, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it) 
  {
	  InputPixelType c = it.Get();
	  if (c < -1000) 
    {
	    it.Set (-1000);
    }
  }
}

int
main(int argc, char *argv[])
{
  
    if (argc != 4 && ( argc != 7 ) ) 
    {
	std::cerr << "Missing Parameters " << std::endl;
	std::cerr << "Usage: " << argv[0] <<",arc="<<argc;
	std::cerr << " InputFileName OutPutFileName typeConvertOption retioX retioY retioZ "<< std::endl;
	std::cerr << " typeConvertOption can be: -u2s, -s2u, -s2s, -u2u, -f2s, -s2uNoAdjust"<< std::endl;
	return 1;
    }
  
    /*
      DicomReaderType::Pointer fixed_input_rdr;
      DicomReaderType::Pointer moving_input_rdr;
      fixed_input_rdr = load_dicom_dir_rdr(argv[1]);
      fixed_input_rdr->Update();
    */
    typedef itk::ImageFileWriter < SignedImageType > SignedWriterType;
    typedef itk::ImageFileWriter < UnsignedImageType > UnsignedWriterType;
    typedef itk::ImageFileWriter < FloatImageType > FloatWriterType;

    int has_resample = 0;
    if (argc > 5) {
	has_resample = 1;
    }

#define TYPE_UNSIGNED    1
#define TYPE_SIGNED      2
#define TYPE_FLOAT       3

    int noadjust = 0;
    int input_type = 0;
    int output_type = 0;
    if (0 == strcmp(argv[3], "-u2s")) {
	input_type = TYPE_UNSIGNED;
	output_type = TYPE_SIGNED;
    } else if (0 == strcmp(argv[3], "-u2u")) {
	input_type = TYPE_UNSIGNED;
	output_type = TYPE_UNSIGNED;
    } else if (0 == strcmp(argv[3] , "-s2s")) {
	input_type = TYPE_SIGNED;
	output_type = TYPE_SIGNED;
    } else if (0 == strcmp(argv[3], "-s2u")) {
	input_type = TYPE_SIGNED;
	output_type = TYPE_UNSIGNED;
    } else if (0 == strcmp(argv[3] , "-f2s")) {
	input_type = TYPE_FLOAT;
	output_type = TYPE_SIGNED;
    } else if (0 == strcmp(argv[3] , "-s2uNoAdjust")) {
	input_type = TYPE_SIGNED;
	output_type = TYPE_UNSIGNED;
	noadjust = 1;
    } else {
	printf ("Error.  Wrong command line parameters.\n");
	printf ("Hit any key to continue.\n");
	getchar ();
	exit (1);
    }

    if (input_type == TYPE_UNSIGNED) {

	UnsignedDicomReaderType::Pointer fixed_input_rdr
		= UnsignedDicomReaderType::New();
	load_dicom_dir_rdr(fixed_input_rdr, argv[1]);
	
	UnsignedImageType::Pointer input_image 
		= fixed_input_rdr->GetOutput();
	if (has_resample) {
	    subsample_image (input_image, atoi(argv[4]),
			    atoi(argv[5]), atoi(argv[6]), -1200);
	}
	if (output_type == TYPE_UNSIGNED) {
	    UnsignedWriterType::Pointer writer = UnsignedWriterType::New();
	    writer->SetFileName(argv[2]);
	    writer->SetInput(input_image);
	    writer->Update();
	}
	else if (output_type == TYPE_SIGNED) {
	    typedef itk::CastImageFilter <UnsignedImageType,
		    SignedImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();
	    caster->SetInput(input_image);
        
	    //cast to signed and output
	    SignedWriterType::Pointer writer = SignedWriterType::New();

	    writer->SetFileName(argv[2]);
	    writer->SetInput(caster->GetOutput());
	    writer->Update();
	}
    }
    else if (input_type == TYPE_SIGNED) {

	SignedDicomReaderType::Pointer fixed_input_rdr
		= SignedDicomReaderType::New();
	load_dicom_dir_rdr(fixed_input_rdr, argv[1]);

	fixed_input_rdr->Update();
	SignedImageType::Pointer input_image = fixed_input_rdr->GetOutput();

	if (has_resample) {
	    subsample_image (input_image, atoi(argv[4]),
			    atoi(argv[5]), atoi(argv[6]), -1200);
	}
	if (output_type == TYPE_SIGNED) {
	    //just output
	    fix_invalid_pixels( input_image );

	    SignedWriterType::Pointer writer = SignedWriterType::New();
	    writer->SetFileName(argv[2]);
	    writer->SetInput(input_image);
	    writer->Update();

	}
	else if (output_type == TYPE_UNSIGNED) {
	    if (noadjust == 0) {
		fix_invalid_pixels_with_shift (input_image);
	    }

	    typedef itk::CastImageFilter <SignedImageType,
		    UnsignedImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();
	    caster->SetInput(input_image);
        
	    UnsignedWriterType::Pointer writer = UnsignedWriterType::New();

	    writer->SetFileName(argv[2]);
	    writer->SetInput(caster->GetOutput());
	    writer->Update();
	}
    }
    else if (input_type == TYPE_FLOAT) {
	FloatDicomReaderType::Pointer fixed_input_rdr
		= FloatDicomReaderType::New();
	load_dicom_dir_rdr(fixed_input_rdr, argv[1]);
	
	fixed_input_rdr->Update();
	FloatImageType::Pointer input_image = fixed_input_rdr->GetOutput();

	if (has_resample) {
	    subsample_image (input_image, atoi(argv[4]),
			    atoi(argv[5]), atoi(argv[6]), -1200);
	}

	if (output_type == TYPE_SIGNED) {
	    shift_pet_values (input_image);
	    
	    typedef itk::CastImageFilter <FloatImageType,
		    SignedImageType > CastFilterType;
	    CastFilterType::Pointer caster = CastFilterType::New();
	    caster->SetInput(input_image);
        
	    SignedWriterType::Pointer writer = SignedWriterType::New();

	    writer->SetFileName(argv[2]);
	    writer->SetInput(caster->GetOutput());
	    writer->Update();
	}
    }

    printf ("Finished!\n");
    return 0;
}
