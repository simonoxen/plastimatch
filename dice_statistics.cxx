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
	//float sp_test=0;
	//float dice_alt=0;
	int i=0;

	DoubleVectorType c_ref;
	DoubleVectorType c_warp;
	DoubleVectorType median_center;
	double vol_ref;
	double vol_warp;
	double median_vol;
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
	//sp_test=0;
	alpha=0;
	//dice_alt=0;
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
				FN++;
			}
		}
		if(warped->GetPixel(k)){
			sizeWarp++;
			if(warped->GetPixel(k)!=reference->GetPixel(k))
				FP++;
		}
		if(warped->GetPixel(k)==0 && reference->GetPixel(k)==0){
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
//	vol_ref=moment->GetTotalMass();
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
	//median_vol=abs(vol_ref-vol_warp)/2;
	//median_center[0]=abs(c_ref[0]-c_warp[0])/2;
	//median_center[1]=abs(c_ref[1]-c_warp[1])/2;
	//median_center[2]=abs(c_ref[2]-c_warp[2])/2;

	median_vol=(vol_ref+vol_warp)/2;
	median_center[0]=(c_ref[0]+c_warp[0])/2;
	median_center[1]=(c_ref[1]+c_warp[1])/2;
	median_center[2]=(c_ref[2]+c_warp[2])/2;

	percVolOver=(volOver/median_vol)*100;	
	//printf("MEDIAN VOLUME: %f\n", median_vol);
	//printf("MEDIAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	//fprintf(output,"median\t %f\t %f\n",  median_vol, percVolOver);
	//printf("MEDIAN CENTER OF MASS: %g %g %g\n",median_center[0],median_center[1],median_center[2]);
	printf("MEAN VOLUME: %f\n", median_vol);
	printf("MEAN VOLUME OVERLAP PERC: %f \n",percVolOver);
	fprintf(output,"mean\t %f\t %f\n",  median_vol, percVolOver);
	printf("MEAN CENTER OF MASS: %g %g %g\n",median_center[0],median_center[1],median_center[2]);

	fprintf(output,"\n");
	fprintf(output,"CENTER_OF_MASS\n EXPERT\t");
	fprintf(output, "x\t\t y\t\t z\n");
	fprintf(output,"ref\t %g\t %g\t %g\n",c_ref[0],c_ref[1],c_ref[2]);
	fprintf(output,"warp\t %g\t %g\t %g\n",c_warp[0],c_warp[1],c_warp[2]);
	fprintf(output,"mean \t %g\t %g\t %g\n",median_center[0],median_center[1],median_center[2]);
	fprintf(output,"\n");

	//END calculus of volumes, centers of mass
	
	
	//BEGIN calculus DICE + FP,FN,TP,TN

	//scaling of the TN with respect to FP
	if( (float)FP/(float)(FP+TN)<=1){
		//printf("alfa: %f", (double)((FP+TN)/sizeRef));
		alpha=ceil((double)((FP+TN)/sizeRef));
	}
	
	printf("overlap=TP: %d\n",overlap);
	fprintf(output,"TP: %d\n",overlap);
	printf("\n\n");
	fprintf(output,"\n");

	printf("TN: %d\n",TN);
	fprintf(output,"TN: %d\n",TN);
	printf("\n\n");
	fprintf(output,"\n");

	printf("FN: %d\n",FN);
	fprintf(output,"FN: %d\n",FN);
	printf("\n\n");
	fprintf(output,"\n");

	printf("FP: %d\n",FP);
	fprintf(output,"FP: %d\n",FP);
	printf("\n\n");
	fprintf(output,"\n");

	dice=((float)(2*overlap))/((float)(sizeRef+sizeWarp));
	printf("DICE COEFFICIENT: %f\n",dice);
	fprintf(output,"DICE: %f\n",dice);
	printf("\n\n");
	fprintf(output,"\n");

	//float dice2=0;
	//dice2=((float)(2*overlap))/((float)(2*overlap+FN+FP));
	//printf("DICE2 COEFFICIENT: %f\n",dice2);
	//fprintf(output,"DICE2: %f\n",dice2);
	//printf("\n\n");
	//fprintf(output,"\n");

	printf("ALPHA: %f\n",alpha);

	se=((float)overlap)/((float)(overlap+FN));
	printf("Sensitivity: %f\n",se);
	fprintf(output,"Sensitivity: %f\n",se);
	printf("\n\n");
	fprintf(output,"\n");

	if (alpha==0){
		printf("alpha = 0 \n");
		sp=1;
	}else{
		sp=-((float)FP/(float)(alpha*sizeRef))+1;
	}
	printf("SP: %f\n",sp);
	fprintf(output,"SP: %f\n",sp);
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
	DoubleVectorType median_center;
	double vol_ex1;
	double vol_ex2;
	double vol_ex3;
	double median_vol;
	int FPex12=0;
	int FPex13=0;
	int FPex23=0;
	int FNex12=0;
	int FNex13=0;
	int FNex23=0;
	int TNex12=0;
	int TNex13=0;
	int TNex23=0;
	float se12=0;
	float sp12=0;
	float se13=0;
	float sp13=0;
	float se23=0;
	float sp23=0;
	float se21=0;
	float sp21=0;
	float se31=0;
	float sp31=0;
	float se32=0;
	float sp32=0;
	double alpha12;
	double alpha13;
	double alpha23;
	double alpha21;
	double alpha31;
	double alpha32;

	MomentCalculatorType::Pointer moment= MomentCalculatorType::New();
	

	if(ex_1->GetLargestPossibleRegion().GetSize() != ex_2->GetLargestPossibleRegion().GetSize() && ex_1->GetLargestPossibleRegion().GetSize() != ex_3->GetLargestPossibleRegion().GetSize()){
				fprintf(stderr,"ERROR: The 3 volumes have different sizes. \n");
				fprintf(stderr, "Size expert 1: %d %d %d \n ",ex_1->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size expert 2: %d %d %d \n ",ex_2->GetLargestPossibleRegion().GetSize());
				fprintf(stderr, "Size expert 3: %d %d %d \n ",ex_3->GetLargestPossibleRegion().GetSize());
				exit(-1);
	}

	overlapE12=0;
	overlapE13=0;
	overlapE23=0;
	sizeEx_1=0;
	sizeEx_2=0;
	sizeEx_3=0;
	TNex12=FPex12=FNex12=0;
	TNex13=FPex13=FNex13=0;
	TNex23=FPex23=FNex23=0;
	sp12=se12=0;
	sp13=se13=0;
	sp23=se23=0;
	sp21=se21=0;
	sp31=se31=0;
	sp32=se32=0;
	alpha12=alpha13=alpha23=0;

	get_image_header(dim, offset, spacing, ex_1);
				
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
			if(ex_1->GetPixel(k)){
				if(ex_2->GetPixel(k)){
					overlapE12++;
				}else{
					FNex12++;
				}
			}else if(ex_2->GetPixel(k)){
					FPex12++;
			}else if(ex_1->GetPixel(k)==ex_2->GetPixel(k)==0){
				TNex12++;
			}

			if(ex_1->GetPixel(k)){
				if(ex_3->GetPixel(k)){
					overlapE13++;
				}else{
					FNex13++;
				}
			}else if(ex_3->GetPixel(k)){
					FPex13++;
			}else if(ex_1->GetPixel(k)==ex_3->GetPixel(k)==0){
				TNex13++;
			}

			if(ex_2->GetPixel(k)){
				if(ex_3->GetPixel(k)){
					overlapE23++;
				}else{
					FNex23++;
				}
			}else if(ex_3->GetPixel(k)){
					FPex23++;
			}else if(ex_2->GetPixel(k)==ex_3->GetPixel(k)==0){
				TNex23++;
			}
		}else if(ex_1->GetPixel(k)==ex_2->GetPixel(k)==ex_3->GetPixel(k)){
			overlap++;
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

	median_vol=(vol_ex1+vol_ex2+vol_ex3)/3;
	//median_vol=(max(vol_ex1,vol_ex2,vol_ex3)-min(vol_ex1,vol_ex2,vol_ex3))/2;
	median_center[0]=(c_ex1[0]+c_ex2[0]+c_ex3[0])/3;
	median_center[1]=(c_ex1[1]+c_ex2[1]+c_ex3[1])/3;
	median_center[2]=(c_ex1[2]+c_ex2[2]+c_ex3[2])/3;
	//median_center[0]=(max(c_ex1[0],c_ex2[0],c_ex3[0])-min(c_ex1[0],c_ex2[0],c_ex3[0]))/2;
	//median_center[1]=(max(c_ex1[1],c_ex2[1],c_ex3[1])-min(c_ex1[1],c_ex2[1],c_ex3[1]))/2;
	//median_center[2]=(max(c_ex1[2],c_ex2[2],c_ex3[2])-min(c_ex1[2],c_ex2[2],c_ex3[2]))/2;

	//printf("MEDIAN VOLUME: %f\n", median_vol);
	//percVolOver=(volOver/median_vol)*100;
	//fprintf(output,"median\t %f\t %f\n",median_vol,percVolOver);
	//printf("MEDIAN CENTER OF MASS: %g %g %g\n",median_center[0],median_center[1],median_center[2]);
	printf("MEAN VOLUME: %f\n", median_vol);
	percVolOver=(volOver/median_vol)*100;
	fprintf(output,"mean\t %f\t %f\n",median_vol,percVolOver);
	printf("MEAN CENTER OF MASS: %g %g %g\n",median_center[0],median_center[1],median_center[2]);
		
	fprintf(output,"CENTER_OF_MASS\n EXPERT\t");
	fprintf(output, "x\t\t y\t\t z\n");
	fprintf(output,"ex_1\t %g\t %g\t %g\n",c_ex1[0],c_ex1[1],c_ex1[2]);
	fprintf(output,"ex_2\t %g\t %g\t %g\n",c_ex2[0],c_ex2[1],c_ex2[2]);
	fprintf(output,"ex_3\t %g\t %g\t %g\n",c_ex3[0],c_ex3[1],c_ex3[2]);	
	fprintf(output,"mean \t %g\t %g\t %g\n",median_center[0],median_center[1],median_center[2]);
	//fprintf(output,"median \t %g\t %g\t %g\n",median_center[0],median_center[1],median_center[2]);

	//BEGIN calculus DICE + FP,FN,TP,TN

	//scaling of the TN with respect to FP
	if( (float)FPex12/(float)(FPex12+TNex12)<=1)
		alpha12=ceil((double)((FPex12+TNex12)/sizeEx_1));
	if( (float)FPex13/(float)(FPex13+TNex13)<=1)
		alpha13=ceil((double)((FPex13+TNex13)/sizeEx_1));
	if( (float)FPex23/(float)(FPex23+TNex23)<=1)
		alpha23=ceil((double)((FPex23+TNex23)/sizeEx_2));
	if( (float)FNex12/(float)(FNex12+TNex12)<=1)
		alpha21=ceil((double)((FNex12+TNex12)/sizeEx_2));
	if( (float)FNex13/(float)(FNex13+TNex13)<=1)
		alpha31=ceil((double)((FNex13+TNex13)/sizeEx_3));
	if( (float)FNex23/(float)(FNex23+TNex23)<=1)
		alpha32=ceil((double)((FNex23+TNex23)/sizeEx_3));


	printf("overlap12=TP12==TP21: %d\n",overlapE12);
	fprintf(output,"TP12==TP21: %d\n",overlapE12);
	printf("overlap13=TP13==TP31: %d\n",overlapE13);
	fprintf(output,"TP13==TP31: %d\n",overlapE13);
	printf("overlap13=TP23==TP32: %d\n",overlapE23);
	fprintf(output,"TP23==TP32: %d\n",overlapE23);
	printf("overlap: %d\n",overlap);
	fprintf(output,"overlap: %d\n",overlap);
	printf("\n\n");
	fprintf(output,"\n");

	printf("TN12==TN21: %d\n",TNex12);
	fprintf(output,"TN12==TN21: %d\n",TNex12);
	printf("TN13==TN31: %d\n",TNex13);
	fprintf(output,"TN13==TN31: %d\n",TNex13);
	printf("TN23==TN32: %d\n",TNex23);
	fprintf(output,"TN23==TN32: %d\n",TNex23);
	printf("\n\n");
	fprintf(output,"\n");

	printf("FN12==FP21: %d\n",FNex12);
	fprintf(output,"FN12==FP21: %d\n",FNex12);
	printf("FN13==FP31: %d\n",FNex13);
	fprintf(output,"FN13==FP31: %d\n",FNex13);
	printf("FN23==FP32: %d\n",FNex23);
	fprintf(output,"FN23==FP32: %d\n",FNex23);
	printf("\n\n");
	fprintf(output,"\n");

	printf("FP12==FN21: %d\n",FPex12);
	fprintf(output,"FP12==FN21: %d\n",FPex12);
	printf("FP13==FN31: %d\n",FPex13);
	fprintf(output,"FP13==FN31: %d\n",FPex13);
	printf("FP23==FN32: %d\n",FPex23);
	fprintf(output,"FP23==FN32: %d\n",FPex23);
	printf("\n\n");
	fprintf(output,"\n");

	printf("ALPHA12: %f\n",alpha12);
	printf("ALPHA13: %f\n",alpha13);
	printf("ALPHA23: %f\n",alpha23);
	printf("ALPHA21: %f\n",alpha21);
	printf("ALPHA31: %f\n",alpha31);
	printf("ALPHA32: %f\n",alpha32);
	fprintf(output,"ALPHA12: %f\n",alpha12);
	fprintf(output,"ALPHA13: %f\n",alpha13);
	fprintf(output,"ALPHA23: %f\n",alpha23);
	fprintf(output,"ALPHA21: %f\n",alpha21);
	fprintf(output,"ALPHA31: %f\n",alpha31);
	fprintf(output,"ALPHA32: %f\n",alpha32);
	printf("\n\n");
	fprintf(output,"\n");
	
	se12=((float)overlapE12)/((float)overlapE12+FNex12);
	printf("Sensitivity12: %f\n",se12);
	fprintf(output,"Sensitivity12: %f\n",se12);
	se13=((float)overlapE13)/((float)overlapE13+FNex13);
	printf("Sensitivity13: %f\n",se12);
	fprintf(output,"Sensitivity13: %f\n",se13);
	se23=((float)overlapE12)/((float)overlapE23+FNex23);
	printf("Sensitivity23: %f\n",se12);
	fprintf(output,"Sensitivity23: %f\n",se23);
	se21=((float)overlapE12)/((float)overlapE12+FPex12);
	printf("Sensitivity21: %f\n",se21);
	fprintf(output,"Sensitivity21: %f\n",se21);
	se31=((float)overlapE12)/((float)overlapE12+FPex13);
	printf("Sensitivity31: %f\n",se31);
	fprintf(output,"Sensitivity31: %f\n",se31);
	se32=((float)overlapE23)/((float)overlapE23+FPex23);
	printf("Sensitivity32: %f\n",se32);
	fprintf(output,"Sensitivity32: %f\n",se32);
	printf("\n\n");
	fprintf(output,"\n");

	if(alpha12==0){
		sp12=1;
	}else{
		sp12=-((float)FPex12/(float)(alpha12*sizeEx_1))+1;
	}
	printf("SP12: %f\n",sp12);
	fprintf(output,"SP12: %f\n",sp12);
	if(alpha13==0){
		sp13=1;
	}else{
		sp13=-((float)FPex13/(float)(alpha13*sizeEx_1))+1;
	}
	printf("SP13: %f\n",sp13);
	fprintf(output,"SP13: %f\n",sp13);
	if(alpha23==0){
		sp23=1;
	}else{
		sp23=-((float)FPex23/(float)(alpha23*sizeEx_2))+1;
	}
	printf("SP23: %f\n",sp23);
	fprintf(output,"SP23: %f\n",sp23);
	
	if(alpha21==0){
		sp21=1;
	}else{
		sp21=-((float)FNex12/(float)(alpha21*sizeEx_2))+1;
	}
	printf("SP21: %f\n",sp21);
	fprintf(output,"SP21: %f\n",sp21);

	if(alpha31==0){
		sp31=1;
	}else{
		sp31=-((float)FNex13/(float)(alpha31*sizeEx_3))+1;
	}
	printf("SP31: %f\n",sp31);
	fprintf(output,"SP31: %f\n",sp31);

	if(alpha32==0){
		sp32=1;
	}else{
		sp32=-((float)FNex23/(float)(alpha32*sizeEx_3))+1;
	}
	printf("SP32: %f\n",sp32);
	fprintf(output,"SP32: %f\n",sp32);
	printf("\n\n");
	fprintf(output,"\n");

}

