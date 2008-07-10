//===========================================================





//===========================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "slice_extraction.h"


typedef float	PixelType;
typedef itk::Image<PixelType, 3>	ImgType;
typedef itk::Image<PixelType, 2>	intImgType;
typedef itk::ImageRegionIteratorWithIndex<ImgType> IteratorType;

void print_usage (void)
{
	printf ("This executable computes the DICE coefficient for 2 give binary volumes \n");
	printf ("Usage: contour_statistics \n");
	printf ("  reference volume\t");
	printf ("  warped volume\t");
	printf ("  mode (options: slice, global)\n");
	exit (-1);
}

int main(int argc, char* argv[])
{
	inImgType::IndexType k;
	k[0]=0;
	int overlap=0;
	float dice=0;

	if (argc<4)
		print_usage();
	ImgType::Pointer reference=load_float(argv[1]);
	ImgType::Pointer warped=load_float(argv[2]);
	int size=0;


	if(reference->GetLargestPossibleRegion().GetSize() != warped->GetLargestPossibleRegion().GetSize()){
		fprintf(stderr,"ERROR: The 2 volumes have different sizes. \n");
		fprintf(stderr, "Size Reference: %d %d %d \n ",reference->GetLargestPossibleRegion().GetSize());
		fprintf(stderr, "Size Warped: %d %d %d \n ",warped->GetLargestPossibleRegion().GetSize());
	}
	
	IteratorType it(reference, reference->GetLargestPossibleRegion());
	
	while(!it.IsAtEnd())
	{
		k=it.GetIndex();
		if(reference->GetPixel(k)==1){
			size++;
			if(warped->GetPixel(k)==reference->GetPixel(k)){
				overlap++;
			}
		}
		it.operator ++();
	}
	printf("overlap: %d\n",overlap);
	printf("# of white pixels in the reference image: %d\n",size);
	dice=(2*overlap)/(2*size);
	printf("DICE COEFFICIENT: %f\n",dice);

}









//printf("index: %d %d %d\n",k);
		//system("PAUSE");
		/*intImgType::Pointer sRef;
		sRef = slice_extraction(reference, k[2]);
		intImgType::Pointer sWarp;
		sWarp = slice_extraction(warped, k[2]);*/