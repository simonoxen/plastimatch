/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageMomentsCalculator.h"
#include "dice_statistics.h"

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
	
    size_t dim[3];
    float offset[3];
    float spacing[3];

    DoubleVector3DType c_ref;
    DoubleVector3DType c_warp;
    double vol_ref;
    double vol_warp;
	

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
template float do_dice<unsigned char>(UCharImageType::Pointer reference, UCharImageType::Pointer warped, FILE* output);
template float do_dice<short>(ShortImageType::Pointer reference, ShortImageType::Pointer warped, FILE* output);
