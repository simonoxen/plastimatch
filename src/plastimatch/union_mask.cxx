/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* This program merges (takes the union) of binary (mask) images */
#include "plm_config.h"
#include <time.h>
#include "itkImageRegistrationMethod.h"
#include "itkMeanSquaresImageToImageMetric.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkImage.h"
#include "itkVersorRigid3DTransform.h"
#include "itkCenteredTransformInitializer.h"
#include "itkVersorRigid3DTransformOptimizer.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkSquaredDifferenceImageFilter.h"
#include "itkDICOMImageIO2.h"
#include "itkImageSeriesReader.h"
#include "itkDICOMSeriesFileNames.h"
#include "itkImageFileWriter.h"
#include "itkCommand.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkAffineTransform.h"
#include "itkTranslationTransform.h"
#include "itkRegularStepGradientDescentOptimizer.h"
#include "itkAmoebaOptimizer.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkImageRegionIterator.h"

/* We only deal with these kinds of images... */
const unsigned int Dimension = 3;

typedef unsigned char MaskPixelType;
typedef itk::Image < unsigned char, Dimension > MaskImageType;
typedef itk::ImageFileReader < MaskImageType > MaskReaderType;

MaskReaderType::Pointer
load_mha_rdr(char *fn)
{
    MaskReaderType::Pointer reader = MaskReaderType::New();
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

void
merge_pixels (MaskImageType::Pointer im_out, 
	      MaskImageType::Pointer im_1, 
	      MaskImageType::Pointer im_2)
{
    typedef itk::ImageRegionIterator< MaskImageType > IteratorType;
    MaskImageType::RegionType r_1 = im_1->GetLargestPossibleRegion();
    MaskImageType::RegionType r_2 = im_2->GetLargestPossibleRegion();

    const MaskImageType::PointType& og = im_1->GetOrigin();
    const MaskImageType::SpacingType& sp = im_1->GetSpacing();
    
    im_out->SetRegions(r_1);
    im_out->SetOrigin(og);
    im_out->SetSpacing(sp);
    im_out->Allocate();

    IteratorType it_1 (im_1, r_1);
    IteratorType it_2 (im_2, r_2);
    IteratorType it_out (im_out, r_2);

    for (it_1.GoToBegin(); !it_1.IsAtEnd(); ++it_1,++it_2,++it_out) {
	MaskPixelType p1 = it_1.Get();
	MaskPixelType p2 = it_2.Get();
	it_out.Set (p1 | p2);
    }
}

int
main (int argc, char *argv[])
{
  
    if (argc != 4) {
	std::cerr << "Missing Parameters " << std::endl;
	std::cerr << "Usage: " << argv[0] <<",arc="<<argc;
	std::cerr << " input_1 input_2 output"<< std::endl;
	return 1;
    }

    typedef itk::ImageFileWriter < MaskImageType > MaskWriterType;

    MaskImageType::Pointer im_1 = MaskImageType::New();
    MaskImageType::Pointer im_2 = MaskImageType::New();
    MaskImageType::Pointer im_out = MaskImageType::New();
    im_1 = load_mha_rdr(argv[1])->GetOutput();
    im_2 = load_mha_rdr(argv[2])->GetOutput();

    merge_pixels (im_out, im_1, im_2);

    MaskWriterType::Pointer writer = MaskWriterType::New();
    writer->SetFileName(argv[3]);
    writer->SetInput(im_out);
    writer->Update();

    printf ("Finished!\n");
    getchar();
    return 0;
}
