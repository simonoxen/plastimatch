//===========================================================





//===========================================================

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itk_image.h"
#include "slice_extraction.h"
//#include "itkImageLinearIteratorWithIndex.h"


/* =======================================================================*
    Definitions
 * =======================================================================*/

//typedef itk::ImageFileReader<inImgType>	ReaderType;
typedef itk::ExtractImageFilter<inImgType,outImgType>	FilterType;
//typedef itk::ImageLinearConstIteratorWithIndex<outImgType>	ConstIteratorType;
typedef itk::ImageFileWriter<outImgType>	WriterType;

outImgType::Pointer slice_extraction(inImgType::Pointer reader, int index)
{

	FilterType::Pointer extraction=FilterType::New();
	
	try
	{
		reader->Update(); 
		//std::cout << "Ho letto!" << std::endl;
	}
	catch ( itk::ExceptionObject &err)
	{
		std::cout << "ExceptionObject caught a !" << std::endl; 
		std::cout << err << std::endl; 
		//return -1;
	}
	
	inImgType::RegionType inputRegion=reader->GetLargestPossibleRegion();
	inImgType::SizeType size = inputRegion.GetSize();
	size[2] = 0;

	/*inImgType::SizeType slices = reader->GetLargestPossibleRegion().GetSize();
	std::cout<< slices << std::endl;*/
	inImgType::IndexType start = inputRegion.GetIndex(); 
	start[2]=index;

	inImgType::RegionType desiredRegion; 
	desiredRegion.SetSize(size);
	desiredRegion.SetIndex(start);

	extraction->SetExtractionRegion(desiredRegion);
	extraction->SetInput(reader);


	//	outImgType::ConstPointer outImg = outImgType::New();
	outImgType::Pointer outImg = outImgType::New();
	
	try
	{
		extraction->Update();
		outImg = extraction->GetOutput();
	}
	catch ( itk::ExceptionObject &err)
	{
		std::cout << "ExceptionObject caught a !" << std::endl; 
		std::cout << err << std::endl; 
		//return -1;
	}

	//ConstIteratorType inputIt(outImg,outImg->GetRequestedRegion());
	
	//inputIt.SetDirection(0);
	

	/*for ( inputIt.GoToBegin(); !inputIt.IsAtEnd(); inputIt.NextLine())
	{	
		inputIt.GoToBeginOfLine();
	

		while ( ! inputIt.IsAtEndOfLine() )
		{
			printf("%1.0f",inputIt.Get() );
			++inputIt;
		}
		printf ("\n");
	}*/
	//save_image(outImg, name);
	
	//WriterType::Pointer writer = WriterType::New();
	//writer->SetFileName("goofy.mha");
	//std::cout << writer->GetFileName() << std::endl;

	//writer->SetInput(outImg);
	//std::cout << "Set input of writer" << std::endl;
	//std::cout << writer->GetInput() << std::endl;
	//try
	//{
	//	writer->Update();
	//}
	//catch ( itk::ExceptionObject &err)
	//{
	//	std::cout << "ExceptionObject caught !" << std::endl; 
	//	std::cout << err << std::endl; 
	//	//return -1;   
	//}

	return outImg;
}
