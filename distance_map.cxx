#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#ifdef __BORLANDC__
#define ITK_LEAN_AND_MEAN
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "itk_image.h"
//#include "itkDanielssonDistanceMapImageFilter.h"
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"

typedef UCharImageType InputImageType;
//typedef FloatImageType OutputImageType; 
typedef FloatImageType OutputImageType; 
typedef itk::ImageFileReader< InputImageType  >  ReaderType;
typedef itk::ImageFileReader< OutputImageType  >  ReaderDistanceType;
typedef itk::ImageFileWriter< OutputImageType >  WriterType;
typedef itk::SignedMaurerDistanceMapImageFilter< InputImageType, OutputImageType >  FilterType;
typedef itk::SubtractImageFilter< OutputImageType, OutputImageType, OutputImageType > DifferenceFilterType;

void print_usage (void)
{
	std::cerr << "Usage: distance_map inputImageFile1 outputDistanceMapImageFile1 " <<std::endl;
	std::cerr << std::endl;
	std::cerr << "[inputImageFile2 outputDistanceMapImageFile2 DifferenceImage]"<<std::endl;
    std::cerr << std::endl;
	std::cerr << "This function computes the distance map of the input file(s) according to Maurer implementation" << std::endl;
	std::cerr << "Only Binary inputs are allowed. The inside of the contours are going to have negative distance values" << std::endl;
	exit (-1);
}

void compute_distance_map(char* infn, char* outfn){
	WriterType::Pointer writer = WriterType::New();
	FilterType::Pointer filter = FilterType::New();
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName( infn );
	writer->SetFileName( outfn );
	filter->SetSquaredDistance( false ); 
	filter->SetUseImageSpacing( false ); 
  
	filter->SetInput(reader->GetOutput());
	filter->Update();
	writer->SetInput( filter->GetOutput() );
	writer->Update();
}

void compare_distance_map(char* img1fn ,char* img2fn,char* outfn){
	  DifferenceFilterType::Pointer difference = DifferenceFilterType::New();
	  WriterType::Pointer writer2 = WriterType::New();
	  ReaderDistanceType::Pointer r1 = ReaderDistanceType::New();
	  ReaderDistanceType::Pointer r2 = ReaderDistanceType::New();

	  r1->SetFileName(img1fn);
	  r2->SetFileName(img2fn);
  
	  difference->SetInput1( r1->GetOutput() );
	  difference->SetInput2( r2->GetOutput() );
	  writer2->SetFileName( outfn );
      writer2->SetInput( difference->GetOutput() ); 
	  writer2->Update();


}


int main( int argc, char * argv[] )
{
	InputImageType::Pointer img=InputImageType::New();
	//ImgType::Pointer warped=ImgType::New();

	char* inputImageFile1Name;
	char* inputImageFile2Name;
	char* outputDistanceMap1Name;
	char* outputDistanceMap2Name;
	char* outputDifferenceImageName;
  
	if( argc < 3 ){
	  print_usage();
	}else if (argc <4){
		inputImageFile1Name  = argv[1];    
		outputDistanceMap1Name = argv[2];
		//img=itk_image_load_uchar (inputImageFile1Name, 0);
		//compute_distance_map(img, outputDistanceMap1Name);
		compute_distance_map(inputImageFile1Name, outputDistanceMap1Name);

	}else{
		inputImageFile1Name  = argv[1];    
		outputDistanceMap1Name = argv[2];
		inputImageFile2Name = argv[3];
		outputDistanceMap2Name = argv[4];
		outputDifferenceImageName = argv[5];
		compute_distance_map(inputImageFile1Name, outputDistanceMap1Name);
		std::cout << "saved first output" << std::endl;
		compute_distance_map(inputImageFile2Name, outputDistanceMap2Name);
		std::cout << "saved second output" << std::endl;
		compare_distance_map(outputDistanceMap1Name,outputDistanceMap2Name,outputDifferenceImageName);
		std::cout << "saved difference output" << std::endl;

	}

	






}
