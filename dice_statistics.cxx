/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
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
#include "slice_extract.h"
#include "contour_statistics.h"
#include "itkImageMomentsCalculator.h"

typedef itk::ImageRegionIteratorWithIndex<ImgType> ItTypeVolPixel;
typedef itk::ImageRegionIteratorWithIndex<intImgType> ItTypeSlicePixel;
typedef itk::ImageSliceConstIteratorWithIndex<ImgType> ItSliceType;
typedef itk::ImageMomentsCalculator<ImgType> MomentCalculatorType;


void do_dice_global(ImgType::Pointer reference, ImgType::Pointer warped, FILE* output)
{
	ImgType::IndexType k;
	k[0]=0;
	int overlap=0;
	float dice=0;
	int sizeRef=0;
	int sizeWarp=0;
	int FP=0;
	int FN=0;
	int TN=0;
	
	float volOver;
	float percVolOver;
	int dim[3];
	float offset[3];
	float spacing[3];
	float se=0;
	float sp=0;
	float sp_test=0;
	float dice_alt=0;
	int i=0;

	DoubleVectorType c_ref;
	DoubleVectorType c_warp;
	DoubleVectorType mean_center;
	double vol_ref;
	double vol_warp;
	double mean_vol;
	double alpha;
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
	TN=FP=FN=0;
	sp=se=0;
	sp_test=0;
	alpha=0;
	dice_alt=0;
	get_image_header(dim, offset, spacing, reference);

	ItTypeVolPixel it(reference, reference->GetLargestPossibleRegion());

	while(!it.IsAtEnd())
	{
		k=it.GetIndex();
		if(reference->GetPixel(k)){
			sizeRef++;
			if(warped->GetPixel(k)==reference->GetPixel(k)){
				overlap++;
			}else if (warped->GetPixel(k)!=reference->GetPixel(k)){
				sizeWarp++;
				FN++;
			}
		}else if(warped->GetPixel(k)){
			sizeWarp++;
			FP++;
		}else{
			TN++;
		}
		it.operator ++();
		i++;
	}


	
	printf("# of white pixels in the 2 images: REF: %d WARP: %d\n",sizeRef,sizeWarp);

	volOver=overlap*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);

	//computes moments for reference image
	moment->SetImage(reference);
	moment->Compute();
	c_ref=moment->GetCenterOfGravity();
	//vol_ref=moment->GetTotalMass();
	vol_ref=sizeRef*(reference->GetSpacing()[0]*reference->GetSpacing()[1]*reference->GetSpacing()[2]);
	percVolOver=(volOver/vol_ref)*100;

	printf("VOLUME ref: %f\n", vol_ref);
	printf("VOLUME OVERLAP PERC ex_1: %f \n",percVolOver);
	printf("CENTER ref: %g %g %g\n",c_ref[0],c_ref[1],c_ref[2]);


	fprintf(output,"EXPERT\t");
	fprintf(output,"VOLUME\t");
	fprintf(output,"VOL over perc \n");
	fprintf(output,"ref\t %f\t %f\n", vol_ref,percVolOver);

	printf("\n\n");


	//computes moments for warped image
	moment->SetImage(warped);
	moment->Compute();
	c_warp=moment->GetCenterOfGravity();
	//vol_warp=moment->GetTotalMass();
	vol_warp=sizeWarp*(warped->GetSpacing()[0]*warped->GetSpacing()[1]*warped->GetSpacing()[2]);
	percVolOver=(volOver/vol_warp)*100;

	printf("VOLUME warp: %f\n", vol_warp);
	printf("VOLUME OVERLAP PERC ex_2: %f \n",percVolOver);
	printf("CENTER warp: %g %g %g\n",c_warp[0],c_warp[1],c_warp[2]);
	fprintf(output,"warp\t %f\t %f\n", vol_warp, percVolOver);
	printf("\n");
	

	//Writes the overlap volume
	printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);

	mean_vol=(vol_ref+vol_warp)/2;
	mean_center[0]=(c_ref[0]+c_warp[0])/2;
	mean_center[1]=(c_ref[1]+c_warp[1])/2;
	mean_center[2]=(c_ref[2]+c_warp[2])/2;

	percVolOver=(volOver/mean_vol)*100;	

	printf("MEAN VOLUME: %f\n", mean_vol);
	printf("MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	fprintf(output,"mean\t %f\t %f\n",  mean_vol, percVolOver);

	printf("MEAN CENTER OF MASS: %g %g %g\n",mean_center[0],mean_center[1],mean_center[2]);

	fprintf(output,"\n");
	fprintf(output,"CENTER_OF_MASS\n EXPERT\t");
	fprintf(output, "x\t\t y\t\t z\n");
	fprintf(output,"ref\t %g\t %g\t %g\n",c_ref[0],c_ref[1],c_ref[2]);
	fprintf(output,"warp\t %g\t %g\t %g\n",c_warp[0],c_warp[1],c_warp[2]);
	fprintf(output,"mean \t %g\t %g\t %g\n",mean_center[0],mean_center[1],mean_center[2]);
	fprintf(output,"\n");

	//END calculus of volumes, centers of mass
	//BEGIN calculus DICE + FP,FN,TP,TN

	//scaling of the TN with respect to FP
	alpha=ceil((double)(FP/vol_ref));
	
	printf("overlap=TP: %d\n",overlap);
	fprintf(output,"TP: %f\n",overlap);
	printf("\n\n");
	fprintf(output,"\n");

	printf("TN: %d\n",TN);
	fprintf(output,"TN: %f\n",TN);
	printf("\n\n");
	fprintf(output,"\n");

	printf("FN: %d\n",FN);
	fprintf(output,"FN: %f\n",FN);
	printf("\n\n");
	fprintf(output,"\n");

	printf("FP: %d\n",FP);
	fprintf(output,"FP: %f\n",FP);
	printf("\n\n");
	fprintf(output,"\n");

	dice=((float)2*overlap)/((float)(sizeRef+sizeWarp));
	printf("DICE COEFFICIENT: %f\n",dice);
	fprintf(output,"DICE: %f\n",dice);
	printf("\n\n");
	fprintf(output,"\n");

	//dice_alt=((float)2*overlap)/((float)(2*overlap+FP+FN));
	//printf("ALTERNATIVE CALC DICE COEFFICIENT: %f\n",dice_alt);
	//fprintf(output,"ALTERNATIVE DICE: %f\n",dice_alt);
	//printf("\n\n");
	//fprintf(output,"\n");

	printf("ALPHA: %f\n",alpha);

	se=((float)overlap)/((float)overlap+FN);
	printf("Sensitivity: %f\n",se);
	fprintf(output,"Sensitivity: %f\n",se);
	printf("\n\n");
	fprintf(output,"\n");

	sp=((float)TN)/((float)TN+FP);
	printf("Specificity: %f\n",sp);
	fprintf(output,"Specificity: %f\n",sp);
	printf("\n\n");
	fprintf(output,"\n");

	sp_test=-((float)FP/(float)(alpha*vol_ref))+1;
	printf("SP_TESTING: %f\n",sp_test);
	fprintf(output,"SP_TESTING: %f\n",sp_test);
	printf("\n\n");
	fprintf(output,"\n");
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
	fprintf(output,"DICE: %f\n",dice);
	//fprintf(output,"%f\n",dice);
	fprintf(output,"DICE E1-E2: %f\n",diceE12);
	fprintf(output,"DICE E1-E3: %f\n",diceE13);
	fprintf(output,"DICE E2-E3: %f\n",diceE23);
	printf("\n\n");

	volOver=overlap*(ex_1->GetSpacing()[0]*ex_1->GetSpacing()[1]*ex_1->GetSpacing()[2]);

	//computes moments for first expert
	moment->SetImage(ex_1);
	moment->Compute();
	c_ex1=moment->GetCenterOfGravity();
	vol_ex1=moment->GetTotalMass();
	//vol_ex1=sizeEx_1*(ex_1->GetSpacing()[0]*ex_1->GetSpacing()[1]*ex_1->GetSpacing()[2]);
	percVolOver=(volOver/vol_ex1)*100;

	printf("VOLUME ex_1: %f\n", vol_ex1);
	printf("VOLUME OVERLAP PERC ex_1: %f \n",percVolOver);
	printf("CENTER ex_1: %g %g %g\n",c_ex1[0],c_ex1[1],c_ex1[2]);
	fprintf(output,"EXPERT\t");
	fprintf(output,"VOLUME\t");
	fprintf(output,"VOL over perc\n");
	fprintf(output,"ex_1\t %f\t %f\n", vol_ex1,percVolOver);
	//fprintf(output,"%f\t %f\n", vol_ex1,percVolOver);	


	printf("\n\n");
	fprintf(output,"\n");

	//computes moments for second expert
	moment->SetImage(ex_2);
	moment->Compute();
	c_ex2=moment->GetCenterOfGravity();
	vol_ex2=moment->GetTotalMass();
	//vol_ex2=sizeEx_2*(ex_2->GetSpacing()[0]*ex_2->GetSpacing()[1]*ex_2->GetSpacing()[2]);
	percVolOver=(volOver/vol_ex2)*100;

	printf("VOLUME ex_2: %f\n", vol_ex2);
	printf("VOLUME OVERLAP PERC ex_2: %f \n",percVolOver);
	printf("CENTER ex_2: %g %g %g\n",c_ex2[0],c_ex2[1],c_ex2[2]);

	fprintf(output,"ex_2\t %f\t %f\n", vol_ex2, percVolOver);
	//fprintf(output,"%f\t %f\n", vol_ex2, percVolOver);

	printf("\n\n");

	//computes moments for third expert
	moment->SetImage(ex_3);
	moment->Compute();
	c_ex3=moment->GetCenterOfGravity();
	vol_ex3=moment->GetTotalMass();
	//vol_ex3=sizeEx_3*(ex_3->GetSpacing()[0]*ex_3->GetSpacing()[1]*ex_3->GetSpacing()[2]);
	percVolOver=(volOver/vol_ex3)*100;

	printf("VOLUME ex_3: %f\n", vol_ex3);
	printf("VOLUME OVERLAP PERC ex_3: %f \n",percVolOver);
	printf("CENTER ex_3: %g %g %g\n",c_ex3[0],c_ex3[1],c_ex3[2]);
	
	fprintf(output,"ex_3\t %f\t %f\n", vol_ex3, percVolOver);
	//fprintf(output,"%f\t %f\n", vol_ex3, percVolOver);
	
	printf("\n\n");

	//Writes the overlap volume
	
	printf("VOLUME GLOBAL OVERLAP: %f\n", volOver);
	//fprintf(output,"VOLUME GLOBAL OVERLAP: %f\n", volOver);

	mean_vol=(vol_ex1+vol_ex2+vol_ex3)/3;
	mean_center[0]=(c_ex1[0]+c_ex2[0]+c_ex3[0])/3;
	mean_center[1]=(c_ex1[1]+c_ex2[1]+c_ex3[1])/3;
	mean_center[2]=(c_ex1[2]+c_ex2[2]+c_ex3[2])/3;

	printf("MEAN VOLUME: %f\n", mean_vol);

	percVolOver=(volOver/mean_vol)*100;
	//printf("MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	//fprintf(output,"MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	fprintf(output,"mean\t %f\t %f\n",mean_vol,percVolOver);
	//fprintf(output,"%f\t %f\n",mean_vol,percVolOver);

	printf("MEAN CENTER OF MASS: %g %g %g\n",mean_center[0],mean_center[1],mean_center[2]);
	

	
	fprintf(output,"CENTER_OF_MASS\n EXPERT\t");
	fprintf(output, "x\t\t y\t\t z\n");
	fprintf(output,"ex_1\t %g\t %g\t %g\n",c_ex1[0],c_ex1[1],c_ex1[2]);
	fprintf(output,"ex_2\t %g\t %g\t %g\n",c_ex2[0],c_ex2[1],c_ex2[2]);
	fprintf(output,"ex_3\t %g\t %g\t %g\n",c_ex3[0],c_ex3[1],c_ex3[2]);
	fprintf(output,"mean \t %g\t %g\t %g\n",mean_center[0],mean_center[1],mean_center[2]);
	//fprintf(output,"%g\t %g\t %g\n",c_ex1[0],c_ex1[1],c_ex1[2]);
	//fprintf(output,"%g\t %g\t %g\n",c_ex2[0],c_ex2[1],c_ex2[2]);
	//fprintf(output,"%g\t %g\t %g\n",c_ex3[0],c_ex3[1],c_ex3[2]);
	//fprintf(output,"%g\t %g\t %g\n",mean_center[0],mean_center[1],mean_center[2]);

}

