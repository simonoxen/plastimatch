/*===========================================================
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
===========================================================*/
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
#include "contour_statistics.h"

typedef itk::ImageRegionIteratorWithIndex<ImgType> ItTypeVolPixel;
typedef itk::ImageRegionIteratorWithIndex<intImgType> ItTypeSlicePixel;
typedef itk::ImageSliceConstIteratorWithIndex<ImgType> ItSliceType;


//typedef struct sliceDice SLICEDICE;
//struct sliceDice {
//    int num_slice;
//	int first_slice;
//    float* dice_list;
//};

void do_dice_global(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output)
{
	ImgType::IndexType k;
	//intImgType::IndexType p;
	k[0]=0;
	//p[0]=0;
	int overlap=0;
	float dice=0;
	int size=0;
	//int index=0;
	float volRef;
	float volOver;
	float percVolOver;
	//int volSize=0;
	//int volOverlap=0;
	//SLICEDICE* slice_dice=(SLICEDICE*)malloc(sizeof(SLICEDICE));
	//memset(slice_dice,0,sizeof(SLICEDICE));
	//slice_dice->num_slice=0;
	//slice_dice->first_slice=0;

	

	if(reference->GetLargestPossibleRegion().GetSize() != warped->GetLargestPossibleRegion().GetSize()){
				fprintf(stderr,"ERROR: The 2 volumes have different sizes. \n");
				fprintf(stderr, "Size Reference: %d %d %d \n ",reference->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size Warped: %d %d %d \n ",warped->GetLargestPossibleRegion().GetSize());
				exit(-1);
	}

	//if(strcmp("global",argv[3])==0){
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
			fprintf(output,"DICE COEFFICIENT: %f\n",dice);
			//dim=reference->GetSpacing();
			//volume=size*(dim[0]*dim[1]*dim[2]);
			volRef=size*(reference->GetSpacing()[0]*reference->GetSpacing()[1]*reference->GetSpacing()[2]);
			volOver=overlap*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);
			percVolOver=(volOver/volRef)*100;
			//printf("spacing: %f %f %f\n",reference->GetSpacing()[0],reference->GetSpacing()[1],reference->GetSpacing()[2]);
			printf("VOLUME GLOBAL REFERENCE: %f\n", volRef);
			printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
			printf("VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);

			fprintf(output,"VOLUME GLOBAL REFERENCE: %f\n", volRef);
			fprintf(output,"VOLUME GLOBAL OVERLAP: %f\n", volOver);
			fprintf(output,"VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);

	//}else if(strcmp("slice",argv[3])==0){
}

void do_dice_slice(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output)
{
	ImgType::IndexType k;
	intImgType::IndexType p;
	k[0]=0;
	p[0]=0;
	int overlap=0;
	float dice=0;
	int size=0;
	int index=0;
	float volRef;
	float volOver;
	float percVolOver;
	int volSize=0;
	int volOverlap=0;
	SLICEDICE* slice_dice=(SLICEDICE*)malloc(sizeof(SLICEDICE));
	memset(slice_dice,0,sizeof(SLICEDICE));
	slice_dice->num_slice=0;
	slice_dice->first_slice=0;

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
			sRef = slice_extraction(reference, index, (unsigned char) 0);
			intImgType::Pointer sWarp;
			sWarp = slice_extraction(warped, index, (unsigned char) 0);

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
		fprintf(output,"VOLUME GLOBAL REFERENCE: %f\n", volRef);
		fprintf(output,"VOLUME GLOBAL OVERLAP: %f\n", volOver);
		fprintf(output,"VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);
		fprintf(output,"DICE's COEFFICIENTS\n");

		for(int j=0; j<slice_dice->num_slice; j++)
			fprintf(output,"%f %d\n",slice_dice->dice_list[j],j+slice_dice->first_slice);
		//printf("first slice: %d",slice_dice->first_slice);
}


void do_dice_expert(ImgType::Pointer ex_1, ImgType::Pointer ex_2, ImgType::Pointer ex_3, FILE* output)
{
	ImgType::IndexType k;
	//intImgType::IndexType p;
	k[0]=0;
	//p[0]=0;
	int overlap=0;
	float dice=0;
	int size=0;
	//int index=0;
	float volRef=0;
	float volOver=0;
	float percVolOver=0;
	//int volSize=0;
	//int volOverlap=0;
	//SLICEDICE* slice_dice=(SLICEDICE*)malloc(sizeof(SLICEDICE));
	//memset(slice_dice,0,sizeof(SLICEDICE));
	//slice_dice->num_slice=0;
	//slice_dice->first_slice=0;

	

	if(ex_1->GetLargestPossibleRegion().GetSize() != ex_2->GetLargestPossibleRegion().GetSize() && ex_1->GetLargestPossibleRegion().GetSize() != ex_3->GetLargestPossibleRegion().GetSize()){
				fprintf(stderr,"ERROR: The 3 volumes have different sizes. \n");
				fprintf(stderr, "Size expert 1: %d %d %d \n ",ex_1->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size expert 2: %d %d %d \n ",ex_2->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size expert 3: %d %d %d \n ",ex_3->GetLargestPossibleRegion().GetSize());
				exit(-1);
	}

	//if(strcmp("global",argv[3])==0){
			overlap=0;
			size=0;
						
			ItTypeVolPixel it(ex_1, ex_1->GetLargestPossibleRegion());
		
			while(!it.IsAtEnd())
			{
				k=it.GetIndex();
				if(ex_1->GetPixel(k)==1 || ex_2->GetPixel(k)==1 || ex_3->GetPixel(k)==1){
					size++;
					if(ex_1->GetPixel(k)==ex_2->GetPixel(k) && ex_1->GetPixel(k)==ex_3->GetPixel(k)){
						overlap++;
					}
				}
				it.operator ++();
				//printf("K: %d",k);
			}
			printf("overlap: %d\n",overlap);
			printf("# of white pixels in the 3 images: %d\n",size);
			dice=overlap/size;
			printf("DICE COEFFICIENT: %f\n",dice);
			fprintf(output,"DICE COEFFICIENT: %f\n",dice);
			//dim=reference->GetSpacing();
			//volume=size*(dim[0]*dim[1]*dim[2]);
			volRef=size*(ex_1->GetSpacing()[0]*ex_1->GetSpacing()[1]*ex_1->GetSpacing()[2]);
			volOver=overlap*(ex_1->GetSpacing()[0]*ex_1->GetSpacing()[1]*ex_1->GetSpacing()[2]);
			percVolOver=(volOver/volRef)*100;
			//printf("spacing: %f %f %f\n",reference->GetSpacing()[0],reference->GetSpacing()[1],reference->GetSpacing()[2]);
			printf("VOLUME GLOBAL REFERENCE: %f\n", volRef);
			printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
			printf("VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);
			
			fprintf(output,"VOLUME GLOBAL REFERENCE: %f\n", volRef);
			fprintf(output,"VOLUME GLOBAL OVERLAP: %f\n", volOver);
			fprintf(output,"VOLUME GLOBAL OVERLAP PERC: %f \n",percVolOver);
}