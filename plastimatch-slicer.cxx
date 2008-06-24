#include <iostream>
#include "plastimatch-slicerCLP.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiscreteGaussianImageFilter.h"

/* These are the parameters I deleted from the xml file.
   Just in case I need them later...
<parameters>
  <label>Discrete Gaussian Parameters</label>
  <description>Parameters of the Discrete Gaussian Filter</description>
  <double>
      <name>variance</name>
      <longflag>--variance</longflag>
      <description>Variance ( width of the filter kernel) </description>
      <label>Variance</label>
      <default>0.5</default>
  </double>
</parameters>
*/


int main(int argc, char * argv [])
{
    PARSE_ARGS;

    FILE* fp = fopen ("C:/tmp/plmslc-tmp.parms", "w");
    fprintf (fp,
	     "[GLOBAL]\n"
	     "fixed=%s\n"
	     "moving=%s\n"
	     "xf_out=%s\n"
	     "vf_out=%s\n"
	     "img_out=%s\n\n"
	     "[STAGE]\n"
	     "xform=bspline\n"
	     "optim=lbfgsb\n"
	     "impl=gpuit_cpu\n"
	     "max_its=100\n"
	     "convergence_tol=5\n"
	     "grad_tol=1.5\n"
	     "grid_spac=100 100 100\n"
	     "res=5 5 5\n"
	     "[STAGE]\n"
	     "xform=bspline\n"
	     "optim=lbfgsb\n"
	     "impl=gpuit_cpu\n"
	     "max_its=100\n"
	     "convergence_tol=5\n"
	     "grad_tol=1.5\n"
	     "grid_spac=100 100 100\n"
	     "res=4 4 3\n",
	     plmslc_fixed_volume.c_str(),
	     plmslc_moving_volume.c_str(),
	     "C:/tmp/plmslc-xf.txt",
	     "C:/tmp/plmslc-vf.mha",
	     plmslc_warped_volume.c_str());
    fclose (fp);

    return EXIT_FAILURE;
    
#if defined (commentout)
    std::cout << "Hello Slicer!" << std::endl;
    typedef itk::Image< short, 3 > ImageType;
    typedef itk::ImageFileReader< ImageType >  ReaderType;
    typedef itk::ImageFileWriter< ImageType >  WriterType;
    ReaderType::Pointer reader = ReaderType::New();
    WriterType::Pointer writer = WriterType::New();

    reader->SetFileName (helloSlicerInputVolume.c_str());
    writer->SetFileName (helloSlicerOutputVolume.c_str());

    typedef itk::DiscreteGaussianImageFilter <ImageType, ImageType> FilterType;
    FilterType::Pointer filter = FilterType::New();

    try {  
	filter->SetInput(reader->GetOutput());
	filter->SetVariance(variance);
	writer->SetInput(filter->GetOutput());
	writer->Update();
    }

    catch (itk::ExceptionObject &excep)
    {
	std::cerr << argv[0] << ": exception caught !" << std::endl;
	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
#endif
}
