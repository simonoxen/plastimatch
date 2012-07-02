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
#include "plmsys.h"
#include "distance_map.h"
#include "mabs_subject.h"
#include "mabs_vote.h"
#include "plm_math.h"

class Mabs_vote_private {
public:
    Mabs_vote_private () {
        this->sman = new Mabs_subject_manager;
    }
    ~Mabs_vote_private () {
        delete this->sman;
    }
public:
    Mabs_subject_manager* sman;
    
    FloatImageType::Pointer target;
    FloatImageType::Pointer like0;
    FloatImageType::Pointer like1;
    FloatImageType::Pointer weights;
};

Mabs_vote::Mabs_vote ()
{
    d_ptr = new Mabs_vote_private;
}

Mabs_vote::~Mabs_vote ()
{
    delete d_ptr;
}

void
Mabs_vote::set_fixed_image (
    FloatImageType::Pointer target
)
{
    /* Save a copy of target */
    d_ptr->target = target;

    /* Create a like0 image */
    d_ptr->like0 = FloatImageType::New();
    d_ptr->like0->SetOrigin( target->GetOrigin() );
    d_ptr->like0->SetSpacing( target->GetSpacing() );
    d_ptr->like0->SetDirection( target->GetDirection() );
    d_ptr->like0->SetRegions( target->GetLargestPossibleRegion() );
    d_ptr->like0->Allocate();
    d_ptr->like0->FillBuffer( 0.0 );

    /* Create a like1 image */
    d_ptr->like1 = FloatImageType::New();
    d_ptr->like1->SetOrigin( target->GetOrigin() );
    d_ptr->like1->SetSpacing( target->GetSpacing() );
    d_ptr->like1->SetDirection( target->GetDirection() );
    d_ptr->like1->SetRegions( target->GetLargestPossibleRegion() );
    d_ptr->like1->Allocate();
    d_ptr->like1->FillBuffer( 0.0 );
}

void
Mabs_vote::vote (
    FloatImageType::Pointer atlas_image,
    UCharImageType::Pointer atlas_label
)
{
    double sigma = 50;
    double rho = 1;
  
    // create distance map
    Distance_map dmap;
    dmap.set_input_image (atlas_label);
    dmap.run ();
    FloatImageType::Pointer dmap_image = dmap.get_output_image ();

    /* Create iterators */
    itk::ImageRegionIterator< FloatImageType > target_it (
        d_ptr->target, d_ptr->target->GetLargestPossibleRegion());
    itk::ImageRegionIterator< FloatImageType > atlas_image_it (
        atlas_image, atlas_image->GetLargestPossibleRegion());
    itk::ImageRegionIterator< FloatImageType > like0_it (
        d_ptr->like0, d_ptr->like0->GetLargestPossibleRegion());
    itk::ImageRegionIterator< FloatImageType > like1_it (
        d_ptr->like1, d_ptr->like1->GetLargestPossibleRegion());
    itk::ImageRegionIterator< FloatImageType > dmap_it (
        dmap_image, dmap_image->GetLargestPossibleRegion());

    // These are necessary to normalize the label likelihoods
    const unsigned int wt_scale = 1000;
    double med_diff;
    double value;
    double label_likelihood_0;
    double label_likelihood_1;
    double dmap_value;
    int cnt = 0;
    printf ("\tMABS looping through voxels...\n");fflush(stdout);
    for (atlas_image_it.GoToBegin(),
             dmap_it.GoToBegin(),
             target_it.GoToBegin(),
             like0_it.GoToBegin(),
             like1_it.GoToBegin();
         !target_it.IsAtEnd();
         ++atlas_image_it,
             ++dmap_it,
             ++target_it,
             ++like0_it,
             ++like1_it)
    {
        cnt++;
        
        /* Compute similarity between target and atlas images */
        med_diff = target_it.Get() - atlas_image_it.Get();
        value = exp (-(med_diff * med_diff) / (2.0*sigma*sigma))
            / (M_SQRT2PI * sigma);
    
        /* Compute the chance of being in the structure. */
        /* Nb. we need to check to make sure exp(dmap_value) 
           doesn't overflow.  The actual overflow is at about exp(700) 
           for double, and about exp(85) for float.  But we can be 
           a little more conservative. */
        dmap_value = rho * dmap_it.Get();
        if (dmap_value > 50) {
            label_likelihood_0 = 0;
            label_likelihood_1 = 1;
        } else if (dmap_value > -50) {
            label_likelihood_0 = exp (-rho*dmap_value);
            label_likelihood_1 = exp (+rho*dmap_value);
        } else {
            label_likelihood_0 = 1;
            label_likelihood_1 = 0;
        }

        /* Compute total score, weighted by image similarity */
        double sum = label_likelihood_0 + label_likelihood_1;
        double l0 = (label_likelihood_0 / sum) * value;
        double l1 = (label_likelihood_1 / sum) * value;

        // write to like0, like1
        like0_it.Set (like0_it.Get() + l0*wt_scale);
        like1_it.Set (like1_it.Get() + l1*wt_scale);
    }
    printf ("\tMABS voted with %d voxels\n", cnt);
}    

void
Mabs_vote::normalize_votes ()
{
    /* GCS: I don't understand this */
    const unsigned int wt_scale = 1000;

    /* Create weight image */
    d_ptr->weights = FloatImageType::New();
    d_ptr->weights->SetOrigin (d_ptr->target->GetOrigin());
    d_ptr->weights->SetSpacing (d_ptr->target->GetSpacing());
    d_ptr->weights->SetDirection (d_ptr->target->GetDirection());
    d_ptr->weights->SetRegions (d_ptr->target->GetLargestPossibleRegion());
    d_ptr->weights->Allocate ();
    d_ptr->weights->FillBuffer (0.0);

    /* Create iterators */
    itk::ImageRegionIterator< FloatImageType > like0_it (
        d_ptr->like0, d_ptr->like0->GetLargestPossibleRegion());
    itk::ImageRegionIterator< FloatImageType > like1_it (
        d_ptr->like1, d_ptr->like1->GetLargestPossibleRegion());
    itk::ImageRegionIterator< FloatImageType > weights_it (
        d_ptr->weights, d_ptr->weights->GetLargestPossibleRegion());

    /* Normalize log likelihood */
    int cnt = 0;
    double XX_0 = 0, XX_1 = 0;
    double YY_0 = DBL_MAX, YY_1 = DBL_MAX;
    for (weights_it.GoToBegin(),
             like0_it.GoToBegin(),
             like1_it.GoToBegin();
         !like0_it.IsAtEnd();
         ++weights_it,
             ++like0_it,
             ++like1_it)
    {
        cnt++;
        double l0 = like0_it.Get();
        double l1 = like1_it.Get();
        if (l0 > XX_0) XX_0 = l0;
        if (l1 > XX_1) XX_1 = l1;
        if (l0 < YY_0) YY_0 = l0;
        if (l1 < YY_1) YY_1 = l1;
        double v =  l1 / (l1+l0);
        weights_it.Set (v*wt_scale);
    }
    printf ("looped through %d weights.\n", cnt);
    printf ("\tMAX votes = %g, %g\n", XX_0, XX_1);
    printf ("\tMAX votes = %g, %g\n", YY_0, YY_1);
}

FloatImageType::Pointer
Mabs_vote::get_weight_image ()
{
    return d_ptr->weights;
}

/* -----------------------------------------------------------------
   OLD CODE
   ------------------------------------------------------------------ */
#if defined (commentout)
bool
Mabs_vote::vote (
const Mabs_parms& parms
)
{

  const   unsigned int wt_scale = 1000;
  
  /* make image/structure series */
  std::vector< std::string > trainImageSeries;
  std::vector< std::string > distanceMapSeries;
  printf ("Looping through subjects.\n");
  Mabs_subject* sub = parms.sman->current ();
  while (sub) {
    printf ("-- subject\n");
    printf ("   -- img: %s [%p]\n", sub->img_fn, sub->img);
    printf ("   -- ss : %s [%p]\n", sub->ss_fn, sub->ss);
    trainImageSeries.push_back(sub->img_fn);
    distanceMapSeries.push_back(sub->ss_fn);
    
    sub = parms.sman->next ();
  }

  Plm_image target_image (parms.labeling_input_fn);
  target_image.convert(PLM_IMG_TYPE_ITK_DOUBLE);
  ImageType::Pointer input_image = target_image.m_itk_double;
  
  // create like0, like1
  ImageType::Pointer like0_image = ImageType::New();
  like0_image->SetOrigin( input_image->GetOrigin() );
  like0_image->SetSpacing( input_image->GetSpacing() );
  like0_image->SetDirection( input_image->GetDirection() );
  like0_image->SetRegions( input_image->GetLargestPossibleRegion() );
  like0_image->Allocate();
  like0_image->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > like0_it( like0_image, like0_image->GetLargestPossibleRegion() );
  
  ImageType::Pointer like1_image = ImageType::New();
  like1_image->SetOrigin( input_image->GetOrigin() );
  like1_image->SetSpacing( input_image->GetSpacing() );
  like1_image->SetDirection( input_image->GetDirection() );
  like1_image->SetRegions( input_image->GetLargestPossibleRegion() );
  like1_image->Allocate();
  like1_image->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > like1_it( like1_image, like1_image->GetLargestPossibleRegion() );

  // tmp likelihoods
  ImageType::Pointer like0_tmp = ImageType::New();
  like0_tmp->SetOrigin( input_image->GetOrigin() );
  like0_tmp->SetSpacing( input_image->GetSpacing() );
  like0_tmp->SetDirection( input_image->GetDirection() );
  like0_tmp->SetRegions( input_image->GetLargestPossibleRegion() );
  like0_tmp->Allocate();
  like0_tmp->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > like0_tmp_it( like0_tmp, like0_tmp->GetLargestPossibleRegion() );
  
  ImageType::Pointer like1_tmp = ImageType::New();
  like1_tmp->SetOrigin( input_image->GetOrigin() );
  like1_tmp->SetSpacing( input_image->GetSpacing() );
  like1_tmp->SetDirection( input_image->GetDirection() );
  like1_tmp->SetRegions( input_image->GetLargestPossibleRegion() );
  like1_tmp->Allocate();
  like1_tmp->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > like1_tmp_it( like1_tmp, like1_tmp->GetLargestPossibleRegion() );

  ImageType::Pointer weights = ImageType::New();
  weights->SetOrigin( input_image->GetOrigin() );
  weights->SetSpacing( input_image->GetSpacing() );
  weights->SetDirection( input_image->GetDirection() );
  weights->SetRegions( input_image->GetLargestPossibleRegion() );
  weights->Allocate();
  weights->FillBuffer( 0.0 );
  itk::ImageRegionIterator< ImageType > weights_it( weights, weights->GetLargestPossibleRegion() );

  int cnt = 0;
  for (unsigned int i = 0; i < trainImageSeries.size(); i++) {
    //std::string train_file_name = trainImageSeries[i];
    printf ("Using atlas: %s\n",trainImageSeries[i].c_str());
    Plm_image atlas_image (trainImageSeries[i].c_str());
    //std::string dist_map_name = distanceMapSeries[i];
    Plm_image atlas_structure (distanceMapSeries[i].c_str());

    // get this atlas's vote contribution
    vote_contribution(target_image,
                      atlas_image,
                      atlas_structure,
                      like0_tmp,
                      like1_tmp);

    // for each pixel, add it to total likelihood
    cnt = 0;
    for ( like0_it.GoToBegin(),
              like1_it.GoToBegin(),
              like1_tmp_it.GoToBegin(),
              like1_tmp_it.GoToBegin();
          !like0_it.IsAtEnd();
          ++like0_it,
              ++like1_it,
              ++like0_tmp_it,
              ++like1_tmp_it)
    {

      cnt++;
      
      like0_it.Set( like0_it.Get() + like0_tmp_it.Get() );
      like1_it.Set( like1_it.Get() + like1_tmp_it.Get() );
      
    }
    printf ("\tlooped through %d pixels (again).\n", cnt);
  }

  // computing final normalized likelihood
  cnt = 0;
  for ( weights_it.GoToBegin(),
            like0_it.GoToBegin(),
            like1_it.GoToBegin();
        !like0_it.IsAtEnd();
        ++weights_it,
            ++like0_it,
            ++like1_it)
  {
    cnt++;
    double l0 = like0_it.Get();
    double l1 = like1_it.Get();
    double v =  l1 / (l1+l0);
    weights_it.Set(v*wt_scale);
  }
  printf ("looped through %d weights.\n", cnt);
  
  return write_to_file(weights, parms.labeling_output_fn);
}

bool
Mabs_vote::vote_old (
const Mabs_parms& parms
)
{
  printf ("Entering VOTE.\n");
  const   unsigned int wt_scale = 1000;
  Mabs_subject* sub = parms.sman->current ();
  // this->output_fn = parms.labeling_output_fn;
  
  double sigma = 200;
  double rho = 5;
  
  ReaderType::Pointer inputReader = ReaderType::New();
  ReaderType::Pointer trainingReader = ReaderType::New();
  ReaderType::Pointer distmapReader = ReaderType::New();

  /* make image/structure series */
  std::vector< std::string > trainImageSeries;
  std::vector< std::string > distanceMapSeries;
  printf ("Looping through subjects.\n");
  while (sub) {
    printf ("-- subject\n");
    printf ("   -- img: %s [%p]\n", sub->img_fn, sub->img);
    printf ("   -- ss : %s [%p]\n", sub->ss_fn, sub->ss);
    trainImageSeries.push_back(sub->img_fn);
    distanceMapSeries.push_back(sub->ss_fn);
    
    sub = parms.sman->next ();
  }

  unsigned int numberTraining = trainImageSeries.size();
  printf ("Training subjects: %d\n", numberTraining);
  
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

  // PLM_IMG_TYPE_ITK_UCHAR
  Plm_image inputImage_plmi (parms.labeling_input_fn);
  inputImage_plmi.convert(PLM_IMG_TYPE_ITK_DOUBLE);
  ImageType::Pointer inputImage = inputImage_plmi.m_itk_double;
  
  // ImageType::Pointer inputImage = inputReader->GetOutput();
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
    std::cout << "file name: " << train_file_name << std::endl;
    trainingReader->SetFileName( train_file_name );
    try
    {
      trainingReader->Update();
    }
    catch( itk::ExceptionObject& err )
    {
      std::cout << "EXCEPTION" << std::endl;
      std::cout << "Could not read the training image: " << train_file_name << std::endl;
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
      double value = exp( -pow( (med_diff), 2.0 ) / (2.0*pow(sigma, 2.0)) ) / (sqrt(2.0*M_PI)*sigma);
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
    weightsIt.Set(v*wt_scale);
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
#endif
