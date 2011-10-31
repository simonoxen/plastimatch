/*=========================================================================
Program: ITK nSIFT Implemention - Command Line Wrapper
Module: $RCSfile: testnD.cxx,v $
Language: C++
Date: $Date: 2007/11/25 15:51:48 $
Version: $Revision: 1.0 $
Copyright (c) 2005,2006,2007 Warren Cheung
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
* The name of the Insight Consortium, nor the names of any consortium members,
nor of any contributors, may be used to endorse or promote products derived
from this software without specific prior written permission.
* Modified source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
=========================================================================*/

#define VERBOSE
//#define DEBUG
//#define DEBUG_VERBOSE

#define SIFT_FEATURE
//#define REORIENT

#define GENERATE_KEYS 
//#define SUBSAMPLE
//#define CROP
//#define DO_DOUBLE

// Command Line Arguments
int ARG_IMG1=2;
int ARG_IMG2=3;

#include <cstddef>
#include <itkScaleInvariantFeatureImageFilter.h>
#include <itkImageSeriesReader.h>
#include <itkNumericSeriesFileNames.h>
#include <itkAffineTransform.h>
#include <getopt.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkResampleImageFilter.h>

#define DIMENSION 3

int main( int argc, char *argv[] )
{
  const int Dimension = DIMENSION;
  

  // Default scale is 1.0
  double test_scale = 1.0;
  float test_rotate = 0.0;  // 0 degrees
  float test_translate = 0.0; //0 mm
  double test_crop = 0.8;
  //  float test_rotate = 0.0874;  // 5 degrees
  //float test_rotate = 0.1748;  // 10 degrees
  int series_start = 1;
  int series_end = 9;
  int series_inc = 1;

  int mode = 'i';  /* defaults to comparing 2 images */;
  int rotate_middle=0; /* defaults to rotating around the origin */

#define OPT_SCALE 'x'
#define OPT_ROTATE 'r'
#define OPT_TRANSLATE 't'
#define OPT_DIM 'd'
#define OPT_CROP 'c'
#define OPT_SERIES_START 's'
#define OPT_SERIES_END 'e'
#define OPT_SERIES_INC 'i'

  /* new Getopt code */
  while(1) {
    static struct option long_options[] =
      {
	/* These options set a flag.   */
	{"synthetic", 0, &mode, 's'},
	{"image", 0, &mode, 'i'},
	{"rotate-middle", 0, &rotate_middle, 1},
	/* These options don't set a flag.
	   We distinguish them by their indices.  */
	{"scale", required_argument, 0, OPT_SCALE},
	{"rotate", required_argument, 0, OPT_ROTATE},
	{"translate", required_argument, 0, OPT_TRANSLATE},
	{"crop", required_argument, 0, OPT_CROP},
	{"series-start", required_argument, 0, OPT_SERIES_START},
	{"series-end", required_argument, 0, OPT_SERIES_END},
	{"series-inc", required_argument, 0, OPT_SERIES_INC},
	//	{"dimension", required_argument, 0, OPT_DIM},
	{0, 0, 0, 0}
      };    

    int optindex;
    int val = getopt_long(argc, argv, "", long_options, &optindex);

    if (val == -1)
      break;

	bool rotation=0;
	bool translation=0;

    switch(val) {
    case OPT_SCALE:
      test_scale = atof(optarg);
      break;
    case OPT_ROTATE:
      if (atof(optarg) >= 0.0 && atof(optarg) <= 360.0)
      test_rotate = atof(optarg) * PI * 2.0 / 360.0;
	  rotation=1;
      break;
	case OPT_TRANSLATE:
		test_translate = atof(optarg);
		translation=1;
		break;
    case OPT_CROP:
      if (atof(optarg) <= 1.0) {
	test_crop = atof(optarg);
      }
      break;
    case OPT_SERIES_START:
      series_start = atoi(optarg);
      break;
    case OPT_SERIES_END:
      series_end = atoi(optarg);
      break;
    case OPT_SERIES_INC:
      series_inc = atoi(optarg);
      break;
      /*
    case OPT_DIM:
      Dimension = atoi(optarg);
      break;
      */
    }
  }

  ARG_IMG1 = optind;
  ARG_IMG2 = optind+1;

  typedef  float  PixelType;
  typedef itk::Image< PixelType, Dimension >  FixedImageType;
  typedef itk::ScaleInvariantFeatureImageFilter<FixedImageType, Dimension> SiftFilterType;

  typedef itk::ImageSource< FixedImageType > ImageSourceType;

  ImageSourceType::Pointer fixedImageReader, fixedImageReader2;
 
  if( argc <= ARG_IMG1 || (mode == 'i' && argc <= ARG_IMG2))
    {
      std::cerr << "Incorrect number of parameters " << std::endl;
      std::cerr << std::endl;
//      std::cerr << "siftkeys program ( ";
//      std::cerr << DIMENSION << "D ";      
//#ifdef SIFT_FEATURE
//      std::cerr << "sift-feature ";
//#else
//      std::cerr << "histogram-feature ";
//#endif
//#ifdef REORIENT
//      std::cerr << "reoriented ";
//#endif
//      std::cerr << ")" << std::endl;

      std::cerr << "Usage: \n";
      std::cerr  << argv[0] << " [options] ImageFile [ImageFile2]\n"; 
	  std::cerr << "This program takes an input image file(s) and generates scale invariant features." << std::endl;
      std::cerr << std::endl;
      std::cerr << "Image Processing Options (Choose ONE):" << std::endl;
      std::cerr << "--image" << std::endl;
      std::cerr << " compare ImageFile and ImageFile2" << std::endl;
      std::cerr << "\nOR\n" << std::endl;
      std::cerr << "--synthetic" << std::endl;
      std::cerr << " compare ImageFile to synthetically generated version" << std::endl;
      std::cerr << "Synthetic Image Options:" << std::endl;
      std::cerr << "--rotate value" << std::endl;
      std::cerr << "  rotate synthetic image on first axis by value degrees" << std::endl;
	  std::cerr << "--translate value" <<std::endl;
	  std::cerr << " translate synthetic image [mm]" << std::endl;
      std::cerr << "--rotate-middle" << std::endl;
      std::cerr << "  centre of rotation at the centre of image (defaults to origin)" << std::endl;
      std::cerr << "--scale value" << std::endl;
      std::cerr << "  scale all axes of synthetic image by value" << std::endl;
      std::cerr << std::endl;
      return 1;
    }

  std::cerr << "Dimension = " << Dimension << "\n";
  std::cerr << "Test Scale = " << test_scale << "\n";
  std::cerr << "Test Rotate = " << test_rotate << "\n";
  std::cerr << "Test Translate = " << test_translate << "\n";
  std::cerr << "Image Crop Ratio (first 3D) = " << test_crop << "\n";
  std::cerr << "Mode = " << (char) mode << "\n";
  std::cerr << "ImageFile1 = " << argv[optind] << "\n";

  if (Dimension == 4) {
    std::cerr << "Image Series Start = " << series_start << "\n";
    std::cerr << "Image Series End = " << series_end << "\n";
    std::cerr << "Image Series Inc = " << series_inc << "\n";
  }

//#ifdef SIFT_FEATURE
  std::cerr << "SIFT Feature\n" << std::endl;
//#else
//  std::cerr << "Histogram Feature\n" << std::endl;
//#endif
//
//#ifdef REORIENT
//  std::cerr << "Reorientation enabled\n" << std::endl;  
//#endif

  if (Dimension == 4) {
    /* Assume fileseries reader */
    typedef itk::ImageSeriesReader< FixedImageType  > FixedImageReaderType;
    FixedImageReaderType::Pointer tmpImageReader  = FixedImageReaderType::New();

    typedef itk::NumericSeriesFileNames NameGeneratorType;
    NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();
    
    nameGenerator->SetSeriesFormat( argv[ARG_IMG1] );
    nameGenerator->SetStartIndex( series_start );
    nameGenerator->SetEndIndex( series_end );
    nameGenerator->SetIncrementIndex( series_inc );
        
    tmpImageReader->SetFileNames( nameGenerator->GetFileNames() );

    fixedImageReader = tmpImageReader;
  } else {
    typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
    FixedImageReaderType::Pointer tmpImageReader  = FixedImageReaderType::New();
    tmpImageReader  = FixedImageReaderType::New();

    tmpImageReader->SetFileName(  argv[ARG_IMG1] );
    fixedImageReader=tmpImageReader;
  }
  fixedImageReader->Update();

  typedef itk::AffineTransform<double,Dimension> ScaleType;
#ifdef CROP
  ScaleType::Pointer no_transform = ScaleType::New();
  no_transform->SetIdentity();
  
  SiftFilterType::ResampleFilterType::Pointer cropper = SiftFilterType::ResampleFilterType::New();
  cropper->SetInput(fixedImageReader->GetOutput());
  FixedImageType::SizeType cropsize = 
    fixedImageReader->GetOutput()->GetLargestPossibleRegion().GetSize();
std::cout<<"crop "<< cropsize[0]<<" " <<cropsize[1]<<std::endl;
  for (int k = 0; k < Dimension; ++k) {
    if (k < 4)
      cropsize[k] = (int) (cropsize[k] * test_crop);
  }
  cropper->SetSize( cropsize );
  cropper->SetOutputSpacing(fixedImageReader->GetOutput()->GetSpacing());
  cropper->SetTransform(no_transform);
  cropper->Update();
  FixedImageType::Pointer fixedImage = cropper->GetOutput();
#else
  FixedImageType::Pointer fixedImage= FixedImageType::New();
  try{
  fixedImage = fixedImageReader->GetOutput();
          }
        catch (itk::ExceptionObject &err)
        {
            std::cout << "ExceptionObject caught !" << std::endl;
            std::cout << err << std::endl;
            return -1;
        }

#endif

  SiftFilterType::PointSetTypePointer keypoints1, keypoints2;

  SiftFilterType siftFilter1, siftFilter2;
  

#ifdef DEBUG
  siftFilter1.writeImage(fixedImage, "tmp0.png");
  std::cout << std::endl << "Starting SIFT Feature Extraction...\n";  
#endif
  //siftFilter1.writeImage(fixedImage, "provaMAIN.mha");
  if (mode=='s')
	  siftFilter1.SetInitialSigma(1.5);
  if (mode == 'i')
	  siftFilter1.SetInitialSigma(2);

  keypoints1 = siftFilter1.getSiftFeatures(fixedImage, "physicalcoord_max1.fcsv","physicalcoord_min1.fcsv","imagecoord_max1.txt","imagecoord_min1.txt","point_rej_contrast1.fcsv","point_rej_curvature1.fcsv");

  typedef itk::AffineTransform< double, Dimension > TestTransformType;
  typedef    TestTransformType::InputVectorType             VectorType;  
  typedef    TestTransformType::ParametersType              ParametersType;
  TestTransformType::Pointer test_transform = TestTransformType::New();
  TestTransformType::Pointer inv_test_transform = TestTransformType::New();
  test_transform->SetIdentity();
  FixedImageType::Pointer scaledImage = FixedImageType::New();
  
  // Synthetic test image
  if (mode=='s') {
    std::cerr << std::endl << "Synthetic image mode\n";  
   /* bool rotation;
	bool translation;
	if (test_rotate!=0)
	{
		rotation=1;
	}else{
		test_rotate=0.0;
		rotation=0;
	}
	if (test_translate!=0)
	{
		translation=1;
	}else{
		test_translate=0.0;
		translation=0;
	}*/


	const unsigned int np = test_transform->GetNumberOfParameters();

    ParametersType parameters( np ); // Number of parameters
 //   inv_test_transform = TestTransformType::New();
 //          
    TestTransformType::InputPointType translate_vector;
	
	FixedImageType::PointType origin = fixedImage->GetOrigin();
	FixedImageType::SpacingType spacing = fixedImage->GetSpacing();
	FixedImageType::SizeType size = fixedImage->GetLargestPossibleRegion().GetSize();
 //           
 //   test_transform->SetIdentity();
 //   test_transform->Scale( 1.0 / test_scale);

	if (rotate_middle) {
      std::cerr << "Transformation centred at middle of image." << std::endl;    
      /* Cycle through each dimension and shift by half*/
      for (int k = 0; k < Dimension; ++k)
		  //translate_vector[k] = (size[k]/2.0);
		translate_vector[k] = origin[k]+(size[k]/2.0)*spacing[k];
      test_transform->SetCenter(translate_vector);
	  std::cout<<"Center of Transformation: "<<translate_vector<<std::endl;
	  std::cout<<"Origin: "<<fixedImage->GetOrigin()<<std::endl;
    } 
	else {
      std::cerr << "Transformation centred at origin." << std::endl;   
	  test_transform->SetCenter( origin );
	  std::cout<<"Center of Transformation: "<<test_transform->GetCenter()<<std::endl;
	  std::cout<<"Origin: "<<fixedImage->GetOrigin()<<std::endl;
	  }
	
	  //-------ROTATION:
	  test_transform->Rotate(0,1,test_rotate);
	
	  //-------TRANSLATION:
	  TestTransformType::OutputVectorType tr;
	  tr[0]=test_translate;
	  tr[1]= test_translate;
	  tr[2]=test_translate;
	  std::cout << "prova_trasla" << std::endl;
	  test_transform->Translate(tr);

	  //--------SCALING:
	  TestTransformType::OutputVectorType scaling;
	  scaling[0]=scaling[1]=scaling[2]= test_scale;
	  test_transform->Scale(scaling);

	  std::cout << "Transform Parms: " << std::endl;
	  std::cout << test_transform->GetParameters() << std::endl;
	  std::cout << "MATRIX: " << std::endl;
	  std::cout << test_transform->GetMatrix() << std::endl;

	  FixedImageType::Pointer scaledImage;
	  typedef itk::ResampleImageFilter<FixedImageType,FixedImageType> ResampleFilterType;
      ResampleFilterType::Pointer scaler = ResampleFilterType::New();
      scaler->SetInput(fixedImage);
	  //scaler->SetSize(size);
	  //scaler->SetOutputSpacing(spacing);
	  //scaler->SetOutputOrigin(origin);
	  //scaler->SetOutputDirection( fixedImage->GetDirection() );
      
	  FixedImageType::SizeType newsize;
	  FixedImageType::PointType offset;
	  	  
      for (int k = 0; k < Dimension; ++k)
	  		newsize[k] = (unsigned int) size[k] / test_scale;
				
      scaler->SetSize( newsize );
	  std::cout << "NEW SIZE: " << newsize << std::endl;
	  
      scaler->SetOutputSpacing(spacing);
	  	  
	  if(newsize!=size & rotate_middle)  //scaling centred at middle of image
	  {
		  for (int k = 0; k < Dimension; ++k)
			offset[k]=translate_vector[k]-(newsize[k]/2.0)*spacing[k];
		  std::cout<<"New Origin: "<<offset<<std::endl;
		  scaler->SetOutputOrigin(offset);
	  }
	  else
	  {scaler->SetOutputOrigin(origin);}

	  scaler->SetOutputDirection( fixedImage->GetDirection() );

	  //interpolate
	  typedef itk::LinearInterpolateImageFunction< FixedImageType, double >  InterpolatorType;
	  InterpolatorType::Pointer interpolator = InterpolatorType::New();

	 /* typedef itk::BSplineInterpolateImageFunction< FixedImageType, double >  InterpolatorType;
	  InterpolatorType::Pointer interpolator = InterpolatorType::New();
	  interpolator->SetSplineOrder(3);
	  interpolator->SetInputImage(fixedImage);
	  interpolator->UseImageDirectionOn();*/

	  scaler->SetInterpolator( interpolator );
	  scaler->SetDefaultPixelValue( (PixelType) -1200 );
	  scaler->SetTransform(test_transform);
	  //scaler->SetOutputParametersFromImage(fixedImage);
	  scaler->Update();
	  scaledImage = scaler->GetOutput();
      
#ifdef DEBUG
      siftFilter2.writeImage(scaledImage, "tmp1.png");
      std::cout << std::endl;
#endif
	  //FixedImageType::Pointer scaledImage;
	  siftFilter2.SetInitialSigma(1.5);
	  siftFilter2.writeImage(scaledImage, "image_transform.mha");
	  keypoints2 = siftFilter2.getSiftFeatures(scaledImage,"physicalcoord_max2.fcsv","physicalcoord_min2.fcsv","imagecoord_max2.txt","imagecoord_min2.txt","point_rej_contrast2.fcsv","point_rej_curvature2.fcsv");  
    
    
    std::cerr << "Test Image Scale: " << test_scale << std::endl;
    std::cerr << "Test Translate: " << test_translate << std::endl;
    std::cerr << "Test Image Rotate: " << test_rotate << std::endl;    
  } else if (mode == 'i') {
    std::cerr << std::endl << "Image Comparison mode\n";  
    inv_test_transform = NULL;

    if (Dimension == 4) {
      typedef itk::ImageSeriesReader< FixedImageType  > FixedImageReaderType;
      FixedImageReaderType::Pointer tmpImageReader  = FixedImageReaderType::New();
      tmpImageReader  = FixedImageReaderType::New();

      typedef itk::NumericSeriesFileNames NameGeneratorType;
      NameGeneratorType::Pointer nameGenerator = NameGeneratorType::New();
      
      nameGenerator->SetSeriesFormat( argv[ARG_IMG2] );
      nameGenerator->SetStartIndex( series_start );
      nameGenerator->SetEndIndex( series_end );
      nameGenerator->SetIncrementIndex( series_inc );
      
      tmpImageReader->SetFileNames( nameGenerator->GetFileNames() );
      fixedImageReader2 = tmpImageReader;
    } else {      
      typedef itk::ImageFileReader< FixedImageType  > FixedImageReaderType;
      FixedImageReaderType::Pointer tmpImageReader  = FixedImageReaderType::New();
      tmpImageReader  = FixedImageReaderType::New();

      tmpImageReader->SetFileName(  argv[ARG_IMG2] );
      fixedImageReader2 = tmpImageReader;
    }
    fixedImageReader2->Update();

#ifdef CROP
    SiftFilterType::ResampleFilterType::Pointer cropper2 = SiftFilterType::ResampleFilterType::New();
    cropper2->SetInput(fixedImageReader2->GetOutput());
    FixedImageType::SizeType cropsize2 = 
      fixedImageReader2->GetOutput()->GetLargestPossibleRegion().GetSize();
    for (int k = 0; k < Dimension; ++k) {
      if (k < 4)
	cropsize2[k] /= 2;
    }
    cropper2->SetSize( cropsize2 );
    cropper2->SetOutputSpacing(fixedImageReader2->GetOutput()->GetSpacing());
    cropper2->SetTransform(no_transform);
    cropper2->Update();
    FixedImageType::Pointer fixedImage2 = cropper2->GetOutput();
#else
    FixedImageType::Pointer fixedImage2 = fixedImageReader2->GetOutput();  
#endif
	siftFilter2.SetInitialSigma(2);
    keypoints2 = siftFilter2.getSiftFeatures(fixedImage2,"physicalcoord_max2.fcsv","physicalcoord_min2.fcsv","imagecoord_max2.txt","imagecoord_min2.txt","point_rej_contrast2.fcsv","point_rej_curvature2.fcsv");
  }
  
  std::cerr << std::endl << "Matching Keypoints\n";  
  siftFilter2.MatchKeypointsPos(keypoints1, keypoints2, inv_test_transform);
#ifdef GENERATE_KEYS
  siftFilter2.MatchKeypointsFeatures(keypoints1, keypoints2, inv_test_transform);
#endif
  return 0;

}


