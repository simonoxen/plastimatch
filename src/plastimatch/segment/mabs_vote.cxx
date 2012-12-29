/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsegment_config.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itkImageRegionIterator.h"

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
    d_ptr->like0->SetOrigin(target->GetOrigin());
    d_ptr->like0->SetSpacing(target->GetSpacing());
    d_ptr->like0->SetDirection(target->GetDirection());
    d_ptr->like0->SetRegions(target->GetLargestPossibleRegion());
    d_ptr->like0->Allocate();
    d_ptr->like0->FillBuffer(0.0);

    /* Create a like1 image */
    d_ptr->like1 = FloatImageType::New();
    d_ptr->like1->SetOrigin(target->GetOrigin());
    d_ptr->like1->SetSpacing(target->GetSpacing());
    d_ptr->like1->SetDirection(target->GetDirection());
    d_ptr->like1->SetRegions(target->GetLargestPossibleRegion());
    d_ptr->like1->Allocate();
    d_ptr->like1->FillBuffer(0.0);
}

void
Mabs_vote::vote (
    FloatImageType::Pointer atlas_image, 
    FloatImageType::Pointer dmap_image
)
{
    double sigma = 50;
    double rho = 1;
  
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
