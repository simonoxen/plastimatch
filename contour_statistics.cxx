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
#include "itkImageSliceConstIteratorWithIndex.h"
#include "slice_extraction.h"


typedef float	PixelType;
typedef itk::Image<PixelType, 3>	ImgType;
typedef itk::Image<PixelType, 2>	intImgType;
typedef itk::ImageRegionIteratorWithIndex<ImgType> ItTypeVolPixel;
typedef itk::ImageRegionIteratorWithIndex<intImgType> ItTypeSlicePixel;
typedef itk::ImageSliceConstIteratorWithIndex<ImgType> ItSliceType;


typedef struct sliceDice SLICEDICE;
struct sliceDice {
    int num_slice;
	int first_slice;
    float* dice_list;
};
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
	ImgType::IndexType k;
	intImgType::IndexType p;
	k[0]=0;
	p[0]=0;
	int overlap=0;
	float dice=0;
	int size=0;
	//int num_sl;
	int index=0;
	//float dim[3];
	float volRef;
	float volOver;
	float percVolOver;
	int volSize=0;
	int volOverlap=0;
	SLICEDICE* slice_dice=(SLICEDICE*)malloc(sizeof(SLICEDICE));
	memset(slice_dice,0,sizeof(SLICEDICE));
	slice_dice->num_slice=0;
	slice_dice->first_slice=0;

	if (argc<4)
		print_usage();
	ImgType::Pointer reference=load_float(argv[1]);
	ImgType::Pointer warped=load_float(argv[2]);
	

	if(reference->GetLargestPossibleRegion().GetSize() != warped->GetLargestPossibleRegion().GetSize()){
				fprintf(stderr,"ERROR: The 2 volumes have different sizes. \n");
				fprintf(stderr, "Size Reference: %d %d %d \n ",reference->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size Warped: %d %d %d \n ",warped->GetLargestPossibleRegion().GetSize());
				exit(-1);
	}
	
	if(strcmp("global",argv[3])==0){
			overlap=0;
			size=0;
						
			ItTypeVolPixel it(reference, reference->GetLargestPossibleRegion());
		
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
			//dim=reference->GetSpacing();
			//volume=size*(dim[0]*dim[1]*dim[2]);
			volRef=size*(reference->GetSpacing()[0]*reference->GetSpacing()[1]*reference->GetSpacing()[2]);
			volOver=overlap*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);
			percVolOver=(volOver/volRef)*100;
			//printf("spacing: %f %f %f\n",reference->GetSpacing()[0],reference->GetSpacing()[1],reference->GetSpacing()[2]);
			printf("VOLUME GLOBAL REFERENCE: %f\n", volRef);
			printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
			printf("VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);

	}else if(strcmp("slice",argv[3])==0){

		printf("You've chosen to compute Dice coefficient for each slice\n");		
		//num_sl=(int)reference->GetLargestPossibleRegion().GetSize()[2];		
		//slice_dice->num_slice=num_sl;
		//printf("num_slices: %d\n",slice_dice->num_slice);
		ItSliceType itSlice(reference, reference->GetLargestPossibleRegion());
		itSlice.SetFirstDirection(0);
		itSlice.SetSecondDirection(1);
		/*slice_dice->dice_list=(float*)malloc(num_sl*sizeof(float));*/

		while(!itSlice.IsAtEnd())
		{
			
			overlap=0;
			size=0;
			k=itSlice.GetIndex();
			index=k[2];
			intImgType::Pointer sRef;
			sRef = slice_extraction(reference, index);
			intImgType::Pointer sWarp;
			sWarp = slice_extraction(warped, index);

			ItTypeSlicePixel iter(sRef, sRef->GetLargestPossibleRegion());
			while(!iter.IsAtEnd())
			{
				p=iter.GetIndex();
				if(sRef->GetPixel(p)==1){
					size++;
					if(sWarp->GetPixel(p)==sRef->GetPixel(p)){
						overlap++;
					}
				}
				iter.operator ++();
				//printf("overlap: %d\n",overlap);
			}
			//printf("overlap: %d\n",overlap);
			//printf("# of white pixels in the reference image: %d\n",size);
			
			if(overlap==0 && size==0){
				//printf("slice %d is full of air. No dice coefficient computed\n", index);
				//system("PAUSE");
				//slice_dice->dice_list[index]=0;
			}else if(overlap!=0 && size==0){
				fprintf(stderr,"Something is wrong: you found overlapping region on a non-existant region\n");
				exit(-1);
			}else{
				volSize=volSize+size;
				volOverlap=volOverlap+overlap;
				if(slice_dice->first_slice==0){
					slice_dice->first_slice=index;
					printf("First contour is on slice %d\n", slice_dice->first_slice);
				}
				slice_dice->num_slice++;
				//printf("SLICE: %d ELEM: %d\n", index, slice_dice->num_slice);
				slice_dice->dice_list=(float*)realloc(slice_dice->dice_list,2*sizeof(float));
				slice_dice->dice_list[slice_dice->num_slice-1]=(2*overlap)/(2*size);
				//printf("coeff: %f\n",slice_dice->dice_list[slice_dice->num_slice-1]);
			}

			itSlice.NextSlice();
		}
		volRef=volSize*(reference->GetSpacing()[0]*reference->GetSpacing()[1]*reference->GetSpacing()[2]);
		volOver=volOverlap*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);
		percVolOver=(volOver/volRef)*100;
		//printf("spacing: %f %f %f\n",reference->GetSpacing()[0],reference->GetSpacing()[1],reference->GetSpacing()[2]);
		printf("VOLUME GLOBAL REFERENCE: %f\n", volRef);
		printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
		printf("VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);
		//for(int j=0; j<slice_dice->num_slice; j++)
			//printf("DICE COEFFICIENT in slice %d: %f\n",j,slice_dice->dice_list[j]);
		//printf("first slice: %d",slice_dice->first_slice);
	}

}