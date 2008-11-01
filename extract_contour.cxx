/*===========================================================
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
===========================================================*/
#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkContourExtractor2DImageFilter.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "slice_extract.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =======================================================================*
    Definitions
 * =======================================================================*/
typedef itk::ContourExtractor2DImageFilter<FloatImage2DType> ContourType;
typedef ContourType::VertexType VertexType;
typedef itk::ImageSliceConstIteratorWithIndex<FloatImageType> IteratorType;

int main(int argc, char ** argv)
{
    FILE* fp;
    FILE* file;
    FloatImageType::IndexType k;
    k[0]=0;

    if( argc < 2 ) {
	printf("Usage: extract_contour input_img [output_file]");
	exit(-1);
    }

    FloatImageType::Pointer volume = load_float(argv[1]);

    IteratorType itSlice (volume, volume->GetLargestPossibleRegion());
    itSlice.SetFirstDirection(0);
    itSlice.SetSecondDirection(1);
	
    if (argc < 3) {
	fp = fopen ("vertices_pixelcoord.txt", "w");
	file = fopen ("vertices_physcoord.txt", "w");
    } else {
	char filename[50]="";
	char filename2[50]="";
	strcpy(filename,argv[2]);
	strcat(filename,"_pixelcoord.txt");
	strcpy(filename2,argv[2]);
	strcat(filename2,"_physcoord.txt");
	fp= fopen(filename,"w");
	file=fopen(filename2,"w");
    }

    if (!fp || !file) { 
	printf ("Could not open vertices file for writing\n");
	return -1;
    }
	

    while(!itSlice.IsAtEnd())
    {
	k=itSlice.GetIndex();
	//printf("%2d\n", k[2]);
		
	FloatImage2DType::Pointer slice;
	slice = slice_extract (volume, k[2], (float) 0.0);

	ContourType::Pointer contour=ContourType::New();

	contour->SetContourValue(0.5);
	contour->SetInput(slice);
		
	try
	{
	    contour->Update();
	    //std::cout << "Cerco il contorno!\n" << std::endl;
	}
	catch (itk::ExceptionObject &err)
	{
	    std::cout << "ExceptionObject caught !" << std::endl; 
	    std::cout << err << std::endl; 
	    return -1;   
	}
		
	//std::cout << "NR OUTPUTS:"<<contour->GetNumberOfOutputs() << std::endl; 
	//system("PAUSE");
	for(unsigned int i = 0; i < contour->GetNumberOfOutputs(); i++)
	{
	    ContourType::VertexListConstPointer vertices =contour->GetOutput(i)->GetVertexList();
	    /*fprintf(fp,"%s %d%s%d\n","Contour",k[2],".",i);*/
	    /*fprintf(fp,"%d%s%d\n",k[2],".",i);*/
	    //fprintf(fp,"\n");
	    fprintf(fp,"%s %s %s\n","NaN","NaN","NaN");
	    fprintf(file,"%s %s %s\n","NaN","NaN","NaN");
	    for(unsigned int j = 0; j < vertices->Size(); j++)
	    {
		const VertexType& vertex = vertices->ElementAt(j);
					
		fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]);
		fprintf(file,"%.3f %.3f %.3f \n",vertex[0]*volume->GetSpacing()[0]+volume->GetSpacing()[3],vertex[1]*volume->GetSpacing()[1]+volume->GetSpacing()[4],k[2]*volume->GetSpacing()[2]+volume->GetSpacing()[5]);
		
		//fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]*volume->GetSpacing()[2]);
		//std::cout << vertex[0] <<" "<<vertex[1]<<" "<<k[2]<<std::endl;
					

	    }
	    //system("PAUSE");
	}
	itSlice.NextSlice();
    }
    fclose(fp);
    fclose(file);

}
