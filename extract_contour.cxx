//===========================================================





//===========================================================
#include "plm_config.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkContourExtractor2DImageFilter.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkImageLinearIteratorWithIndex.h"
#include "slice_extraction.h"
#include "itkImageSliceConstIteratorWithIndex.h"
//#include "itkMetaDataDictionary.h"
//#include "itkMetaDataObject.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =======================================================================*
    Definitions
 * =======================================================================*/

//typedef float	PixelType;
//typedef itk::Image<PixelType, 3>	inImgType;
//typedef itk::Image<PixelType, 2>	outImgType;
//typedef itk::ImageFileWriter<inImgType>	WriterType;
//typedef itk::MetaDataDictionary DictionaryType;
//typedef itk::MetaDataObject< std::string > MetaDataStringType;
typedef itk::ImageFileReader<inImgType>	ReaderType;
typedef itk::ContourExtractor2DImageFilter<outImgType> ContourType;
typedef ContourType::VertexType VertexType;
typedef itk::ImageSliceConstIteratorWithIndex<inImgType> IteratorType;



int main(int argc, char ** argv)
{
    FILE* fp;
    inImgType::IndexType k;
    k[0]=0;

    if( argc < 2 )
    {
		printf("Usage: extract_contour input_img [output_file]");
		exit(-1);
    }

    inImgType::Pointer volume=load_float(argv[1]);

    //std::cout<< "Preparing to load..." << std::endl;

    IteratorType itSlice(volume, volume->GetLargestPossibleRegion());
    itSlice.SetFirstDirection(0);
    itSlice.SetSecondDirection(1);
	
    if(argc <3)
	fp = fopen ("vertices.txt", "w");
    else
	fp= fopen(argv[2],"w");

    if (!fp) { 
	printf ("Could not open vertices file for writing\n");
	return -1;
    }
	

    while(!itSlice.IsAtEnd())
    {
	k=itSlice.GetIndex();
	//printf("%2d\n", k[2]);
		
	outImgType::Pointer slice;
	slice = slice_extraction(volume, k[2]);

	ContourType::Pointer contour=ContourType::New();

	contour->SetContourValue(0.5);
	contour->SetInput(slice);
		
	try
	{
	    contour->Update();
	    //std::cout << "Cerco il contorno!\n" << std::endl;
	}
	catch ( itk::ExceptionObject &err)
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
	    for(unsigned int j = 0; j < vertices->Size(); j++)
	    {
		const VertexType& vertex = vertices->ElementAt(j);
					
		fprintf(fp,"%.3f %.3f %2d\n",vertex[0],vertex[1],k[2]);
		//std::cout << vertex[0] <<" "<<vertex[1]<<" "<<k[2]<<std::endl;
					

	    }
	    //system("PAUSE");
	}
	itSlice.NextSlice();
    }
    fclose(fp);

}





//FILE* fp;
//char buffer[1024];
//float x,y;
//int i=0;
//
//try
//{
//
//	fp = fopen ("goofy.mha", "r");
//	if (!fp) { 
//		printf ("Could not open slice file for read\n");
//		return -1;
//	}
//	for (i=0; i<30; i++) 
//	{ 
//		fgets(buffer,1024,fp);
//		if (strstr(buffer, "ElementSpacing")!=NULL) 
//			sscanf(&(buffer[16]), "%f%f", x, y);
//		printf("%f%f", x, y);
//	}
//}
//catch ( itk::ExceptionObject &err)
//{
//	std::cout << "ExceptionObject caught !" << std::endl; 
//	std::cout << err << std::endl; 
//	return -1;   
//}

//const DictionaryType & dictionary=contour->GetMetaDataDictionary();
//
//	DictionaryType::ConstIterator itr = dictionary.Begin();
//	DictionaryType::ConstIterator end = dictionary.End();
//
//	MetaDataStringType::Pointer value; 
//	while( itr != end )
//	{
//		itk::MetaDataObjectBase::Pointer entry = itr->second;	
//		value= dynamic_cast<MetaDataStringType *>(entry.GetPointer());
//		std::string tag = value->GetMetaDataObjectValue();
//		std::cout<< tag << std::endl;
//		std::string goofy="ElementSpacing";
//		if(tag==goofy)
//		{
//			printf("%s", tag);
//			entry=itr->second;
//			value= dynamic_cast<MetaDataStringType *>(entry.GetPointer());
//			tag=value->GetMetaDataObjectValue();
//			printf("%s", tag);
//			itr=dictionary.End();
//		}
//	}
