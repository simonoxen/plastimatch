/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "plmbase.h"
#include "plmsegment.h"
#include "plmsys.h"

#include "plm_math.h"

Mabs_vote::Mabs_vote ()
{
    this->target_fn[0] = '\0';
    this->output_fn[0] = '\0';
    this->sman = new Mabs_subject_manager;
}

Mabs_vote::~Mabs_vote ()
{
    delete this->sman;
}

bool
Mabs_vote::vote (
const Mabs_parms& parms
)
{
  const   unsigned int wt_scale = 1000000000;
  Mabs_subject* sub = parms.sman->current ();
  // this->output_fn = parms.labeling_output_fn;
  
  double sigma = 200;
  double rho = 5;
  
  ReaderType::Pointer inputReader = ReaderType::New();
  ReaderType::Pointer trainingReader = ReaderType::New();
  ReaderType::Pointer distmapReader = ReaderType::New();
  
  std::vector< std::string > trainImageSeries;
  std::vector< std::string > distanceMapSeries;
  /* make image/structure series */
  unsigned int numberTraining = 0;
  while (sub) {
    fprintf (stderr, "-- subject\n");
    fprintf (stderr, "   -- img: %s [%p]\n", sub->img_fn, sub->img);
    fprintf (stderr, "   -- ss : %s [%p]\n", sub->ss_fn, sub->ss);
    trainImageSeries.push_back(sub->img_fn);
    distanceMapSeries.push_back(sub->ss_fn);
    
    sub = parms.sman->next ();
    numberTraining ++;
  }
  
  /* Load input image we are segmenting */
  inputReader->SetFileName( parms.labeling_input_fn );
  
  try
  {
    inputReader->Update();
  }
  catch( itk::ExceptionObject& err )
  {
    std::cout << "Could not read one of the input images." << std::endl;
    std::cout << err << std::endl;
    return false;
  }
  
  ImageType::Pointer inputImage = inputReader->GetOutput();
  inputImage->DisconnectPipeline();
  itk::ImageRegionConstIterator< ImageType > inputImageIt( inputImage, inputImage->GetLargestPossibleRegion() );
  
  // These store warped training images and distance maps
  ImageType::Pointer trainingImage = NULL;
  ImageType::Pointer distmapImage = NULL;
  
  // Creating images to store the combined likelihoods for each label
  ImageType::Pointer likelihood0_Image = ImageType::New();
  likelihood0_Image->SetOrigin( inputImage->GetOrigin() );
  likelihood0_Image->SetSpacing( inputImage->GetSpacing() );
  likelihood0_Image->SetDirection( inputImage->GetDirection() );
  likelihood0_Image->SetRegions( inputImage->GetLargestPossibleRegion() );
  likelihood0_Image->Allocate();
  likelihood0_Image->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > likelihood0_ImageIt( likelihood0_Image, likelihood0_Image->GetLargestPossibleRegion() );
  
  ImageType::Pointer likelihood1_Image = ImageType::New();
  likelihood1_Image->SetOrigin( inputImage->GetOrigin() );
  likelihood1_Image->SetSpacing( inputImage->GetSpacing() );
  likelihood1_Image->SetDirection( inputImage->GetDirection() );
  likelihood1_Image->SetRegions( inputImage->GetLargestPossibleRegion() );
  likelihood1_Image->Allocate();
  likelihood1_Image->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > likelihood1_ImageIt( likelihood1_Image, likelihood1_Image->GetLargestPossibleRegion() );
  
  ImageType::Pointer like0_Image = ImageType::New();
  like0_Image->SetOrigin( inputImage->GetOrigin() );
  like0_Image->SetSpacing( inputImage->GetSpacing() );
  like0_Image->SetDirection( inputImage->GetDirection() );
  like0_Image->SetRegions( inputImage->GetLargestPossibleRegion() );
  like0_Image->Allocate();
  like0_Image->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > like0It( like0_Image, like0_Image->GetLargestPossibleRegion() );
  
  ImageType::Pointer like1_Image = ImageType::New();
  like1_Image->SetOrigin( inputImage->GetOrigin() );
  like1_Image->SetSpacing( inputImage->GetSpacing() );
  like1_Image->SetDirection( inputImage->GetDirection() );
  like1_Image->SetRegions( inputImage->GetLargestPossibleRegion() );
  like1_Image->Allocate();
  like1_Image->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > like1It( like1_Image, like1_Image->GetLargestPossibleRegion() );
  
  // Final output segmentation image
  ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetOrigin( inputImage->GetOrigin() );
  outputImage->SetSpacing( inputImage->GetSpacing() );
  outputImage->SetDirection( inputImage->GetDirection() );
  outputImage->SetRegions( inputImage->GetLargestPossibleRegion() );
  outputImage->Allocate();
  outputImage->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > outputImageIt( outputImage, outputImage->GetLargestPossibleRegion() );
  
  // Final normalized weights
  ImageType::Pointer weights = ImageType::New();
  weights->SetOrigin( inputImage->GetOrigin() );
  weights->SetSpacing( inputImage->GetSpacing() );
  weights->SetDirection( inputImage->GetDirection() );
  weights->SetRegions( inputImage->GetLargestPossibleRegion() );
  weights->Allocate();
  weights->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > weightsIt( weights, weights->GetLargestPossibleRegion() );
  
  // These are necessary to normalize the label likelihoods
  double labelLikelihood0;
  double labelLikelihood1;
  double normLabelLikelihoods;
  
  // Temporary variable for distmap = 0 fix
  double distmapValue;
  double like0;
  double like1;
  
  // For each training subject
  for (unsigned int i=0; i < numberTraining; i++)
  {
    std::cout << "Processing training subject #: " << i+1 << std::endl;
    
    // Loading warped training image
    std::string train_file_name = trainImageSeries[i];
    trainingReader->SetFileName( train_file_name );
    try
    {
      trainingReader->Update();
    }
    catch( itk::ExceptionObject& err )
    {
      std::cout << "Could not read the training image: " << trainImageSeries[i] << std::endl;
      continue;
    }
    trainingImage = trainingReader->GetOutput();
    trainingImage->DisconnectPipeline();
    
    // Loading distance map of warped training label map
    distmapReader->SetFileName( distanceMapSeries[i] );
    try
    {
      distmapReader->Update();
    }
    catch( itk::ExceptionObject& err )
    {
      std::cout << "Could not read the distance map: " << distanceMapSeries[i] << std::endl;
      continue;
    }
    distmapImage = distmapReader->GetOutput();
    distmapImage->DisconnectPipeline();
    
    // Computing image likelihood (and write it in warped training image)
    int cnt = 0;
    itk::ImageRegionIterator< ImageType > trainingImageIt( trainingImage, trainingImage->GetLargestPossibleRegion() );
    for ( trainingImageIt.GoToBegin(), inputImageIt.GoToBegin(); !trainingImageIt.IsAtEnd(); ++trainingImageIt, ++inputImageIt )
    {
      cnt++;
      
      double med_diff = inputImageIt.Get() - trainingImageIt.Get();
      double value = exp( -pow( (med_diff), 2.0 ) / (2.0*pow(sigma, 2.0)) ) / (sqrt(2.0*PI)*sigma);
      trainingImageIt.Set( value );
    }
    
    std::cout <<  std::endl << "Computing label likelihoods " << cnt << std::endl;

    cnt = 0;
    // Computing label likelihoods
    itk::ImageRegionConstIterator< ImageType > distmapImageIt( distmapImage, distmapImage->GetLargestPossibleRegion() );
    for ( like0It.GoToBegin(), like1It.GoToBegin(), distmapImageIt.GoToBegin(), trainingImageIt.GoToBegin(), likelihood0_ImageIt.GoToBegin(), likelihood1_ImageIt.GoToBegin(); !distmapImageIt.IsAtEnd(); ++distmapImageIt, ++trainingImageIt, ++likelihood0_ImageIt, ++likelihood1_ImageIt,  ++like0It, ++like1It)
    {
      cnt++;
      
      distmapValue = distmapImageIt.Get();
      
      // Fixing locations where distmap is equal to zero
      // (caused by warp to smaller image)
      if ( distmapValue == 0 )
      {
        distmapValue = -50; // -50 value is arbitrary
      }
      
      labelLikelihood0 = exp( -rho*distmapValue );
      labelLikelihood1 = exp( rho*distmapValue );
      normLabelLikelihoods = labelLikelihood0 + labelLikelihood1;
      
      like0 = ( labelLikelihood0 / normLabelLikelihoods ) * trainingImageIt.Get();
      like1 = ( labelLikelihood1 / normLabelLikelihoods ) * trainingImageIt.Get();
      like0It.Set( like0*wt_scale );
      like1It.Set( like1*wt_scale );

      likelihood0_ImageIt.Set( likelihood0_ImageIt.Get() + like0 );
      likelihood1_ImageIt.Set( likelihood1_ImageIt.Get() + like1 );
      
      // std::string like0_File(outputFilename);
      // like0_File = like0_File.substr(0,like0_File.rfind('/'))+"/weights";
      // std::string like_name = train_file_name.substr(train_file_name.rfind('/'));
      // std::string l_suf = "_like0_";
      // l_suf = l_suf+argv[3]+"_"+argv[4];
      // like_name.insert(like_name.rfind('.'), l_suf);
      // like0_File += like_name;
      
      // std::string like1_File(outputFilename);
      // like1_File = like1_File.substr(0,like1_File.rfind('/'))+"/weights";
      // like_name = train_file_name.substr(train_file_name.rfind('/'));
      // l_suf = "_like1_";
      // l_suf = l_suf+argv[3]+"_"+argv[4];
      // like_name.insert(like_name.rfind('.'), l_suf);
      // like1_File += like_name;
      
    }
  }
  
  // Computing final segmentation
  for ( weightsIt.GoToBegin(), outputImageIt.GoToBegin(), likelihood0_ImageIt.GoToBegin(), likelihood1_ImageIt.GoToBegin(); !outputImageIt.IsAtEnd(); ++outputImageIt, ++likelihood0_ImageIt, ++likelihood1_ImageIt, ++weightsIt )
  {
    if ( likelihood1_ImageIt.Get() > likelihood0_ImageIt.Get() )
    {
      outputImageIt.Set( 1 );
    }
    
    double l1 = likelihood1_ImageIt.Get();
    double l0 = likelihood0_ImageIt.Get();
    double v =  l1 / (l1+l0);
    weightsIt.Set(v*1000);
  }
  
  if (write_to_file(outputImage, parms.labeling_output_fn))
  {
    return false;
  }
  
  std::string normFile(parms.labeling_output_fn);
  normFile = normFile.substr(0,normFile.rfind('.'))+"_norm.nii";
  if (write_to_file(weights, normFile))
  {
    return false;
  }
  
  std::cout << std::endl << std::endl;
  return true;
}

void
Mabs_vote::hello_world ()
{
  printf ("hello world");
}


int
Mabs_vote::write_to_file (
    const ImageType::Pointer image_data,
    const std::string out_file
                          )
{
  // Write final segmentation to file
  std::cout << "writing to " << out_file << std::endl;
  
  WriterType::Pointer      writer =  WriterType::New();
  CastFilterType::Pointer  caster =  CastFilterType::New();
  writer->SetFileName( out_file );
  caster->SetInput( image_data );
  
  writer->SetInput( caster->GetOutput() );
  writer->SetUseCompression( true );
  
  try
  {
    writer->Update();
  }
  catch( itk::ExceptionObject& err )
  {
    std::cout << "Unexpected error." << std::endl;
    std::cout << err << std::endl;
    return -1;
  }
  return 0;
}
