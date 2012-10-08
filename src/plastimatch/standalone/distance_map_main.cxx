/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itkSignedMaurerDistanceMapImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkSubtractImageFilter.h"

typedef  unsigned char                   InputPixelType;
typedef  float                 OutputPixelType;
typedef itk::Image< InputPixelType,  3 > InputImageType;
typedef itk::Image< OutputPixelType, 3 > OutputImageType;

//typedef UCharImageType InputImageType;
//typedef FloatImageType OutputImageType; 
//typedef FloatImageType OutputImageType; 
typedef itk::ImageFileReader< InputImageType  >  ReaderType;
typedef itk::ImageFileReader< OutputImageType  >  ReaderDistanceType;
typedef itk::ImageFileWriter< OutputImageType >  WriterType;
typedef itk::SignedMaurerDistanceMapImageFilter< InputImageType, OutputImageType >  FilterType;
typedef itk::SubtractImageFilter< OutputImageType, OutputImageType, OutputImageType > DifferenceFilterType;

void print_usage (void)
{
	std::cerr << "Usage: distance_map inputImageFile1 outputDistanceMapImageFile1 " <<std::endl;
	std::cerr << std::endl;
	std::cerr << "[UseSquaredDistance(1,0) UseImageSpacing(1,0) UseInsideIsPositive(1,0)]"<<std::endl;
    std::cerr << std::endl;
	std::cerr << "[inputImageFile2 outputDistanceMapImageFile2 DifferenceImage]"<<std::endl;
    std::cerr << std::endl;
	std::cerr << "This function computes the distance map of the input file(s) according to Maurer implementation" << std::endl;
	std::cerr << "Only Binary inputs are allowed. Default parameters include squared distance set to false, image spacing is not used and the inside of the contours is set to negative (-> first 3 optional parms set to 0)" << std::endl;
	exit (-1);
}

void compute_distance_map(char* infn, char* outfn, int dist, int img, int inside){
	WriterType::Pointer writer = WriterType::New();
	FilterType::Pointer filter = FilterType::New();
	ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName( infn );
	writer->SetFileName( outfn );
	if(dist==0)
		filter->SetSquaredDistance( false ); 
	else
		filter->SetSquaredDistance( true ); 

	if (img==0)
		filter->SetUseImageSpacing( false ); 
	else
		filter->SetUseImageSpacing( true ); 

	if (inside==0)
		filter->SetInsideIsPositive( false );
	else
		filter->SetInsideIsPositive( true );

	filter->SetNumberOfThreads(2);
  
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
    //InputImageType::Pointer img=InputImageType::New();
    //ImgType::Pointer warped=ImgType::New();

    char* inputImageFile1Name;
    char* inputImageFile2Name;
    char* outputDistanceMap1Name;
    char* outputDistanceMap2Name;
    char* outputDifferenceImageName;
    int dist=0;
    int imgSpac=0;
    int inside=0;
  
    if( argc < 3 ){
        print_usage();
	//}else if (argc <4){
	//	inputImageFile1Name  = argv[1];    
	//	outputDistanceMap1Name = argv[2];
	//	//img=itk_image_load_uchar (inputImageFile1Name, 0);
	//	//compute_distance_map(img, outputDistanceMap1Name);
	//	compute_distance_map(inputImageFile1Name, outputDistanceMap1Name,0,0,0);

    }else{
        inputImageFile1Name  = argv[1];    
        outputDistanceMap1Name = argv[2];
        if (argc<4){
            dist=0;
            imgSpac=0;
            inside=0;
        }else if (argc ==4){
            dist=atoi(argv[3]);
            imgSpac=0;
            inside=0;
        }else if (argc==5){
            dist=atoi(argv[3]);
            imgSpac=atoi(argv[4]);
            inside=0;
        }else if (argc==6){
            dist=atoi(argv[3]);
            imgSpac=atoi(argv[4]);
            inside=atoi(argv[5]);
        }else{
            dist=atoi(argv[3]);
            imgSpac=atoi(argv[4]);
            inside=atoi(argv[5]);
            inputImageFile2Name = argv[6];
            outputDistanceMap2Name = argv[7];
            outputDifferenceImageName = argv[8];
        }

        if(argc<7){
            compute_distance_map(inputImageFile1Name, outputDistanceMap1Name,dist,imgSpac,inside);
        }else{
            compute_distance_map(inputImageFile1Name, outputDistanceMap1Name,dist,imgSpac,inside);	
            std::cout << "saved first output" << std::endl;
            compute_distance_map(inputImageFile2Name, outputDistanceMap2Name,dist,imgSpac,inside);
            std::cout << "saved second output" << std::endl;
            compare_distance_map(outputDistanceMap1Name,outputDistanceMap2Name,outputDifferenceImageName);
            std::cout << "saved difference output" << std::endl;
        }
    }
}
