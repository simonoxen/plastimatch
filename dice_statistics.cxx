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
#include "itkImageMomentsCalculator.h"

typedef itk::ImageRegionIteratorWithIndex<ImgType> ItTypeVolPixel;
typedef itk::ImageRegionIteratorWithIndex<intImgType> ItTypeSlicePixel;
typedef itk::ImageSliceConstIteratorWithIndex<ImgType> ItSliceType;
typedef itk::ImageMomentsCalculator<ImgType> MomentCalculatorType;


//typedef struct sliceDice SLICEDICE;
//struct sliceDice {
//    int num_slice;
//	int first_slice;
//    float* dice_list;
//};

void do_dice_global(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output)
{
	ImgType::IndexType k;
	k[0]=0;
	int overlap=0;
	float dice=0;
	int sizeRef=0;
	int sizeWarp=0;
	float volOver;
	float percVolOver;
	int dim[3];
	float offset[3];
	float spacing[3];
	int i=0;

	DoubleVectorType c_ref;
	DoubleVectorType c_warp;
	DoubleVectorType mean_center;
	double vol_ref;
	double vol_warp;
	double mean_vol;
	MomentCalculatorType::Pointer moment= MomentCalculatorType::New();


	

	if(reference->GetLargestPossibleRegion().GetSize() != warped->GetLargestPossibleRegion().GetSize()){
				fprintf(stderr,"ERROR: The 2 volumes have different sizes. \n");
				fprintf(stderr, "Size Reference: %d %d %d \n ",reference->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size Warped: %d %d %d \n ",warped->GetLargestPossibleRegion().GetSize());
				exit(-1);
	}

	overlap=0;
	sizeRef=0;
	sizeWarp=0;
	get_image_header(dim, offset, spacing, reference);

	//printf("SPACING:%f %f %f\n",spacing[0],spacing[1],spacing[2]);
	//printf("OFFSET: %f %f %f\n",offset[0],offset[1],offset[2]);
	//printf("DIM: %d %d %d\n",dim[0],dim[1],dim[2]);

	ItTypeVolPixel it(reference, reference->GetLargestPossibleRegion());

	while(!it.IsAtEnd())
	{
		k=it.GetIndex();
		//printf("INDICE: %d %d %d",k);
		if(reference->GetPixel(k)){
			sizeRef++;
		}
		if(warped->GetPixel(k)){
			sizeWarp++;
		}
		if(reference->GetPixel(k)|| warped->GetPixel(k)){
			//size++;
			//printf("COORD: %f %f %f \n",x,y,z);
			if(warped->GetPixel(k)==reference->GetPixel(k)){
				overlap++;
			}
		}
		it.operator ++();
		i++;
	}


	printf("overlap: %d\n",overlap);
	printf("# of white pixels in the 2 images: %d\n",sizeRef+sizeWarp);

	dice=((float)2*overlap)/((float)(sizeRef+sizeWarp));
	printf("DICE COEFFICIENT: %f\n",dice);
	fprintf(output,"DICE COEFFICIENT: %f\n",dice);

	printf("\n\n");
	fprintf(output,"\n\n");

	volOver=overlap*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);

	//computes moments for reference image
	moment->SetImage(reference);
	moment->Compute();
	c_ref=moment->GetCenterOfGravity();
	vol_ref=moment->GetTotalMass();
	vol_ref=vol_ref*(reference->GetSpacing()[0]*reference->GetSpacing()[1]*reference->GetSpacing()[2]);
	percVolOver=(volOver/vol_ref)*100;

	printf("VOLUME ex_1: %f\n", vol_ref);
	printf("VOLUME OVERLAP PERC ex_1: %f \n",percVolOver);
	printf("CENTER ex_1: %g %g %g\n",c_ref[0],c_ref[1],c_ref[2]);
	
	fprintf(output,"VOLUME ex_1: %f\n", vol_ref);	
	fprintf(output,"VOLUME OVERLAP PERC ex_1: %f \n",percVolOver);
	fprintf(output,"CENTER OF MASS ex_1: %g %g %g",c_ref[0],c_ref[1],c_ref[2]);

	printf("\n\n");
	fprintf(output,"\n\n");

	//computes moments for warped image
	moment->SetImage(warped);
	moment->Compute();
	c_warp=moment->GetCenterOfGravity();
	vol_warp=moment->GetTotalMass();
	vol_warp=vol_warp*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);
	percVolOver=(volOver/vol_warp)*100;

	printf("VOLUME ex_2: %f\n", vol_warp);
	printf("VOLUME OVERLAP PERC ex_2: %f \n",percVolOver);
	printf("CENTER ex_2: %g %g %g\n",c_warp[0],c_warp[1],c_warp[2]);
	
	fprintf(output,"VOLUME ex_2: %f\n", vol_warp);	
	fprintf(output,"VOLUME OVERLAP PERC ex_2: %f \n",percVolOver);
	fprintf(output,"CENTER OF MASS ex_2: %g %g %g",c_warp[0],c_warp[1],c_warp[2]);
	
	printf("\n\n");
	fprintf(output,"\n\n");

	//Writes the overlap volume
	printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
	fprintf(output,"VOLUME GLOBAL OVERLAP: %f\n", volOver);

	mean_vol=(vol_ref+vol_warp)/2;
	mean_center[0]=(c_ref[0]+c_warp[0])/2;
	mean_center[1]=(c_ref[1]+c_warp[1])/2;
	mean_center[2]=(c_ref[2]+c_warp[2])/2;

	printf("MEAN VOLUME: %f\n", mean_vol);
	fprintf(output,"MEAN VOLUME: %f\n", mean_vol);

	percVolOver=(volOver/mean_vol)*100;
	printf("MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	fprintf(output,"MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);

	printf("MEAN CENTER OF MASS: %g %g %g\n",mean_center[0],mean_center[1],mean_center[2]);
	fprintf(output,"MEAN CENTER OF MASS: %g %g %g\n",mean_center[0],mean_center[1],mean_center[2]);

}

void do_dice_slice(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output)
{
	//this is very buggy...and old!
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
				slice_dice->dice_list[slice_dice->num_slice-1]=((float)2*overlap)/((float)2*size);
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
	k[0]=0;
	int overlapE12=0;
	int overlapE13=0;
	int overlapE23=0;
	int overlap=0;
	float diceE12=0;
	float diceE13=0;
	float diceE23=0;
	float dice=0;
	int sizeEx_1=0;
	int sizeEx_2=0;
	int sizeEx_3=0;
	double volOver=0;
	double percVolOver=0;
	int dim[3];
	float offset[3];
	float spacing[3];
	DoubleVectorType c_ex1;
	DoubleVectorType c_ex2;
	DoubleVectorType c_ex3;
	DoubleVectorType mean_center;
	double vol_ex1;
	double vol_ex2;
	double vol_ex3;
	double mean_vol;
	MomentCalculatorType::Pointer moment= MomentCalculatorType::New();
	

	if(ex_1->GetLargestPossibleRegion().GetSize() != ex_2->GetLargestPossibleRegion().GetSize() && ex_1->GetLargestPossibleRegion().GetSize() != ex_3->GetLargestPossibleRegion().GetSize()){
				fprintf(stderr,"ERROR: The 3 volumes have different sizes. \n");
				fprintf(stderr, "Size expert 1: %d %d %d \n ",ex_1->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size expert 2: %d %d %d \n ",ex_2->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size expert 3: %d %d %d \n ",ex_3->GetLargestPossibleRegion().GetSize());
				exit(-1);
	}

	get_image_header(dim, offset, spacing, ex_1);

	overlapE12=0;
	overlapE13=0;
	overlapE23=0;
	sizeEx_1=0;
	sizeEx_2=0;
	sizeEx_3=0;
				
	ItTypeVolPixel it(ex_1, ex_1->GetLargestPossibleRegion());

	while(!it.IsAtEnd())
	{
		k=it.GetIndex();
		if(ex_1->GetPixel(k)){
			sizeEx_1++;
		}
		if(ex_2->GetPixel(k)){
			sizeEx_2++;
		}
		if(ex_3->GetPixel(k)){
			sizeEx_3++;
		}
		if(ex_1->GetPixel(k) || ex_2->GetPixel(k) || ex_3->GetPixel(k)){
			//size++;
			if(ex_1->GetPixel(k) && ex_2->GetPixel(k)){
				overlapE12++;
			}
			if(ex_1->GetPixel(k)&& ex_3->GetPixel(k)){
				overlapE13++;
			}
			if(ex_2->GetPixel(k)&& ex_3->GetPixel(k)){
				overlapE23++;
			}
			if(ex_2->GetPixel(k)&& ex_3->GetPixel(k) && ex_1->GetPixel(k)){
				overlap++;
			}
		}
		it.operator ++();
		//printf("K: %d",k);
	}

	
	printf("overlap E12: %d\n",overlapE12);
	printf("overlap E13: %d\n",overlapE13);
	printf("overlap E23: %d\n",overlapE23);
	printf("# of white pixels in the ex_1 image: %d\n",sizeEx_1);
	printf("# of white pixels in the ex_2 image: %d\n",sizeEx_2);
	printf("# of white pixels in the ex_3 image: %d\n",sizeEx_3);
	diceE12=((float)2*overlapE12)/((float)(sizeEx_1+sizeEx_2));
	diceE13=((float)2*overlapE13)/((float)(sizeEx_1+sizeEx_3));
	diceE23=((float)2*overlapE23)/((float)(sizeEx_2+sizeEx_3));
	dice=(diceE12+diceE13+diceE23)/3;
	printf("MEAN DICE COEFFICIENT: %f\n",dice);
	fprintf(output,"MEAN DICE COEFFICIENT: %f\n",dice);
	printf("\n\n");
	fprintf(output,"\n\n");
	//dim=reference->GetSpacing();
	//volume=size*(dim[0]*dim[1]*dim[2]);

	volOver=overlap*(ex_1->GetSpacing()[0]*ex_1->GetSpacing()[1]*ex_1->GetSpacing()[2]);

	//computes moments for first expert
	moment->SetImage(ex_1);
	moment->Compute();
	c_ex1=moment->GetCenterOfGravity();
	vol_ex1=moment->GetTotalMass();
	vol_ex1=vol_ex1*(ex_1->GetSpacing()[0]*ex_1->GetSpacing()[1]*ex_1->GetSpacing()[2]);
	percVolOver=(volOver/vol_ex1)*100;

	printf("VOLUME ex_1: %f\n", vol_ex1);
	printf("VOLUME OVERLAP PERC ex_1: %f \n",percVolOver);
	printf("CENTER ex_1: %g %g %g\n",c_ex1[0],c_ex1[1],c_ex1[2]);
	
	fprintf(output,"VOLUME ex_1: %f\n", vol_ex1);	
	fprintf(output,"VOLUME OVERLAP PERC ex_1: %f \n",percVolOver);
	fprintf(output,"CENTER OF MASS ex_1: %g %g %g",c_ex1[0],c_ex1[1],c_ex1[2]);

	printf("\n\n");
	fprintf(output,"\n\n");

	//computes moments for second expert
	moment->SetImage(ex_2);
	moment->Compute();
	c_ex2=moment->GetCenterOfGravity();
	vol_ex2=moment->GetTotalMass();
	vol_ex2=vol_ex2*(ex_2->GetSpacing()[0]*ex_2->GetSpacing()[1]*ex_2->GetSpacing()[2]);
	percVolOver=(volOver/vol_ex2)*100;

	printf("VOLUME ex_2: %f\n", vol_ex2);
	printf("VOLUME OVERLAP PERC ex_2: %f \n",percVolOver);
	printf("CENTER ex_2: %g %g %g\n",c_ex2[0],c_ex2[1],c_ex2[2]);
	
	fprintf(output,"VOLUME ex_2: %f\n", vol_ex2);	
	fprintf(output,"VOLUME OVERLAP PERC ex_2: %f \n",percVolOver);
	fprintf(output,"CENTER OF MASS ex_2: %g %g %g",c_ex2[0],c_ex2[1],c_ex2[2]);

	printf("\n\n");
	fprintf(output,"\n\n");

	//computes moments for third expert
	moment->SetImage(ex_3);
	moment->Compute();
	c_ex3=moment->GetCenterOfGravity();
	vol_ex3=moment->GetTotalMass();
	vol_ex3=vol_ex3*(ex_3->GetSpacing()[0]*ex_3->GetSpacing()[1]*ex_3->GetSpacing()[2]);
	percVolOver=(volOver/vol_ex3)*100;

	printf("VOLUME ex_3: %f\n", vol_ex3);
	printf("VOLUME OVERLAP PERC ex_3: %f \n",percVolOver);
	printf("CENTER ex_3: %g %g %g\n",c_ex3[0],c_ex3[1],c_ex3[2]);
	
	fprintf(output,"VOLUME ex_3: %f\n", vol_ex3);	
	fprintf(output,"VOLUME OVERLAP PERC ex_3: %f \n",percVolOver);
	fprintf(output,"CENTER OF MASS ex_3: %g %g %g",c_ex3[0],c_ex3[1],c_ex3[2]);
	
	printf("\n\n");
	fprintf(output,"\n\n");

	//Writes the overlap volume
	
	printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
	fprintf(output,"VOLUME GLOBAL OVERLAP: %f\n", volOver);

	mean_vol=(vol_ex1+vol_ex2+vol_ex3)/3;
	mean_center[0]=(c_ex1[0]+c_ex2[0]+c_ex3[0])/3;
	mean_center[1]=(c_ex1[1]+c_ex2[1]+c_ex3[1])/3;
	mean_center[2]=(c_ex1[2]+c_ex2[2]+c_ex3[2])/3;

	printf("MEAN VOLUME: %f\n", mean_vol);
	fprintf(output,"MEAN VOLUME: %f\n", mean_vol);

	percVolOver=(volOver/mean_vol)*100;
	printf("MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	fprintf(output,"MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);

	printf("MEAN CENTER OF MASS: %g %g %g\n",mean_center[0],mean_center[1],mean_center[2]);
	fprintf(output,"MEAN CENTER OF MASS: %g %g %g\n",mean_center[0],mean_center[1],mean_center[2]);

}