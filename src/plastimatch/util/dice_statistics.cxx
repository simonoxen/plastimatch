/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageMomentsCalculator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"

#include "compiler_warnings.h"
#include "dice_statistics.h"
#include "itk_image.h"
#include "itk_image_load.h"
#include "logfile.h"
#include "plm_image_header.h"
#include "itk_resample.h"

class Dice_statistics_private {
public:
  Dice_statistics_private () {
    TP = TN = FP = FN = 0;
  }
public:
  size_t TP, TN, FP, FN;
  float dice;
  size_t ref_size;
  size_t cmp_size;
  DoubleVector3DType ref_cog;
  DoubleVector3DType cmp_cog;
  double ref_vol;
  double cmp_vol;
  UCharImageType::Pointer ref_image;
  UCharImageType::Pointer cmp_image;
};

Dice_statistics::Dice_statistics ()
{
  d_ptr = new Dice_statistics_private;
}

Dice_statistics::~Dice_statistics ()
{
  delete d_ptr;
}

void 
Dice_statistics::set_reference_image (const char* image_fn)
{
  d_ptr->ref_image = itk_image_load_uchar (image_fn, 0);
}

void 
Dice_statistics::set_reference_image (
                                      const UCharImageType::Pointer image)
{
  d_ptr->ref_image = image;
}

void 
Dice_statistics::set_compare_image (const char* image_fn)
{
  d_ptr->cmp_image = itk_image_load_uchar (image_fn, 0);
}

void 
Dice_statistics::set_compare_image (
                                    const UCharImageType::Pointer image)
{
  d_ptr->cmp_image = image;
}

void 
Dice_statistics::run ()
{
    /* Resample warped onto geometry of reference */
    if (!itk_image_header_compare (d_ptr->ref_image, d_ptr->cmp_image)) {
        d_ptr->cmp_image = resample_image (d_ptr->cmp_image, 
            Plm_image_header (d_ptr->ref_image), 0, 0);
    }

    /* Initialize counters */
    d_ptr->ref_size = 0;
    d_ptr->cmp_size = 0;
    d_ptr->TP = 0;
    d_ptr->TN = 0;
    d_ptr->FP = 0;
    d_ptr->FN = 0;

    plm_long dim[3];
    float offset[3];
    float spacing[3];
    get_image_header (dim, offset, spacing, d_ptr->ref_image);

    /* Loop through images, gathering Dice */
    itk::ImageRegionIteratorWithIndex<UCharImageType> it (
        d_ptr->ref_image, d_ptr->ref_image->GetLargestPossibleRegion());
    while (!it.IsAtEnd())
    {
        UCharImageType::IndexType k;
        k = it.GetIndex();
        if (d_ptr->ref_image->GetPixel(k)) {
            d_ptr->ref_size++;
            if (d_ptr->cmp_image->GetPixel(k)) {
                d_ptr->TP++;
            } else {
                d_ptr->FN++;
            }
        } else {
            if (d_ptr->cmp_image->GetPixel(k))
                d_ptr->FP++;
            else
                d_ptr->TN++;
        }
        if (d_ptr->cmp_image->GetPixel(k)) {
            d_ptr->cmp_size++;
        }
        ++it;
    }

    /* Do the extra moment stuff */
    typedef itk::ImageMomentsCalculator<UCharImageType> MomentCalculatorType;
    MomentCalculatorType::Pointer moment = MomentCalculatorType::New();

    moment->SetImage (d_ptr->ref_image);
    moment->Compute ();
    d_ptr->ref_cog = moment->GetCenterOfGravity ();
    d_ptr->ref_vol = moment->GetTotalMass ();

    moment->SetImage (d_ptr->cmp_image);
    moment->Compute ();
    d_ptr->cmp_cog = moment->GetCenterOfGravity ();
    d_ptr->cmp_vol = moment->GetTotalMass ();
}

float
Dice_statistics::get_dice ()
{
  return ((float) (2 * d_ptr->TP))
    / ((float) (d_ptr->ref_size + d_ptr->cmp_size));
}

size_t
Dice_statistics::get_true_positives ()
{
  return d_ptr->TP;
}

size_t
Dice_statistics::get_true_negatives ()
{
  return d_ptr->TN;
}

size_t
Dice_statistics::get_false_positives ()
{
  return d_ptr->FP;
}

size_t
Dice_statistics::get_false_negatives ()
{
  return d_ptr->FN;
}

DoubleVector3DType 
Dice_statistics::get_reference_center ()
{
  return d_ptr->ref_cog;
}

DoubleVector3DType 
Dice_statistics::get_compare_center ()
{
  return d_ptr->cmp_cog;
}

double
Dice_statistics::get_reference_volume ()
{
  return d_ptr->ref_vol;
}

double
Dice_statistics::get_compare_volume ()
{
  return d_ptr->cmp_vol;
}


void 
Dice_statistics::debug ()
{
    lprintf ("CENTER_OF_MASS\n");
    lprintf ("ref\t %13g\t %13g\t %13g\n", 
        d_ptr->ref_cog[0], d_ptr->ref_cog[1], d_ptr->ref_cog[2]);
    lprintf ("cmp\t %13g\t %13g\t %13g\n", 
        d_ptr->cmp_cog[0], d_ptr->cmp_cog[1], d_ptr->cmp_cog[2]);

    lprintf ("TP: %13d\n",d_ptr->TP);
    lprintf ("TN: %13d\n",d_ptr->TN);
    lprintf ("FN: %13d\n",d_ptr->FN);
    lprintf ("FP: %13d\n",d_ptr->FP);

    lprintf ("DICE: %13f\n", this->get_dice());
}
