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

#include "plmbase.h"

#include "compiler_warnings.h"
#include "dice_statistics.h"
#include "itk_image.h"

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
    /* Make sure images have the same headers */
    if (!itk_image_header_compare (d_ptr->ref_image, d_ptr->cmp_image)) {
        itkGenericExceptionMacro (
            "ERROR: The 2 volumes have different sizes.\n");
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
		d_ptr->TN++;
	    else
		d_ptr->FP++;
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

template<class T>
float do_dice (
    typename itk::Image<T,3>::Pointer reference, 
    typename itk::Image<T,3>::Pointer warped, 
    FILE* output
)
{
    typedef typename itk::Image<T,3> ImgType;
    typedef typename itk::ImageRegionIteratorWithIndex<ImgType> ItTypeVolPixel;
    typedef typename itk::ImageSliceConstIteratorWithIndex<ImgType> ItSliceType;
    typedef typename itk::ImageMomentsCalculator<ImgType> MomentCalculatorType;

    typename ImgType::IndexType k;
    typename MomentCalculatorType::Pointer moment= MomentCalculatorType::New();

    k[0]=0;
    float dice=0;
    int sizeRef=0;
    int sizeWarp=0;
    int FP=0;
    int FN=0;
    int TN=0;
    int TP=0;
	
    plm_long dim[3];
    float offset[3];
    float spacing[3];

    DoubleVector3DType c_ref;
    DoubleVector3DType c_warp;
    double vol_ref;
    double vol_warp;
    UNUSED_VARIABLE (vol_ref);
    UNUSED_VARIABLE (vol_warp);

    if (reference->GetLargestPossibleRegion().GetSize() 
	!= warped->GetLargestPossibleRegion().GetSize())
    {
	fprintf(stderr,"ERROR: The 2 volumes have different sizes. \n");
	fprintf(stderr, "Size Reference: %lu %lu %lu \n ",
	    reference->GetLargestPossibleRegion().GetSize()[0],
	    reference->GetLargestPossibleRegion().GetSize()[1],
	    reference->GetLargestPossibleRegion().GetSize()[2]);
	fprintf(stderr, "Size Warped: %lu %lu %lu \n ",
	    warped->GetLargestPossibleRegion().GetSize()[0],
	    warped->GetLargestPossibleRegion().GetSize()[1],
	    warped->GetLargestPossibleRegion().GetSize()[2]);
	exit(-1);
    }

    sizeRef=0;
    sizeWarp=0;
    TP=TN=FP=FN=0;
    get_image_header (dim, offset, spacing, reference);

    ItTypeVolPixel it(reference, reference->GetLargestPossibleRegion());

    while (!it.IsAtEnd())
    {
	k=it.GetIndex();
	if(reference->GetPixel(k)){
	    sizeRef++;
	    if(warped->GetPixel(k)==reference->GetPixel(k)){
		TP++;
	    }else if (warped->GetPixel(k)!=reference->GetPixel(k)){
		FN++;
	    }
	}else{
	    if(warped->GetPixel(k)==0)
		TN++;
	    else
		FP++;
	}
	if(warped->GetPixel(k))
	    sizeWarp++;
		
	it.operator ++();
    }


	
    //computes moments for reference image
    moment->SetImage(reference);
    moment->Compute();
    c_ref=moment->GetCenterOfGravity();
    vol_ref=moment->GetTotalMass();

    //printf("VOLUME ref: %f\n", vol_ref);
    //printf("CENTER ref: %g %g %g\n",c_ref[0],c_ref[1],c_ref[2]);
    //printf("\n");


    //computes moments for warped image
    moment->SetImage(warped);
    moment->Compute();
    c_warp=moment->GetCenterOfGravity();
    vol_warp=moment->GetTotalMass();

    //printf("VOLUME warp: %f\n", vol_warp);
    //printf("CENTER warp: %g %g %g\n",c_warp[0],c_warp[1],c_warp[2]);
    //printf("\n");

    fprintf(output,"\n");
    fprintf(output,"CENTER_OF_MASS\n ");
    fprintf(output, "who\t x\t\t y\t\t z\n");
    fprintf(output,"ref\t %g\t %g\t %g\n",c_ref[0],c_ref[1],c_ref[2]);
    fprintf(output,"warp\t %g\t %g\t %g\n",c_warp[0],c_warp[1],c_warp[2]);
    fprintf(output,"\n");

    //END calculus of volumes, centers of mass
	
	
    //BEGIN calculus DICE + FP,FN,TP,TN

		
    //printf("TP=TP: %d\n",TP);
    fprintf(output,"TP: %d\n",TP);
    //printf("\n\n");
    fprintf(output,"\n");

    //printf("TN: %d\n",TN);
    fprintf(output,"TN: %d\n",TN);
    //printf("\n\n");
    fprintf(output,"\n");

    //printf("FN: %d\n",FN);
    fprintf(output,"FN: %d\n",FN);
    //printf("\n\n");
    fprintf(output,"\n");

    //printf("FP: %d\n",FP);
    fprintf(output,"FP: %d\n",FP);
    //printf("\n\n");
    fprintf(output,"\n");

    dice=((float)(2*TP))/((float)(sizeRef+sizeWarp));
    //printf("DICE COEFFICIENT: %f\n",dice);
    fprintf(output,"DICE: %f\n",dice);
    //printf("\n\n");
    fprintf(output,"\n");

    return dice;
}

/* Explicit instantiations */
template 
PLMUTIL_API
float do_dice<unsigned char>(UCharImageType::Pointer reference, UCharImageType::Pointer warped, FILE* output);
#if defined (commentout)
template 
PLMUTIL_API
float do_dice<short>(ShortImageType::Pointer reference, ShortImageType::Pointer warped, FILE* output);
#endif
