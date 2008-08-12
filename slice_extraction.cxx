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
//typedef itk::ImageLinearConstIteratorWithIndex<outImgType>	ConstIteratorType;

template<class T>
typename itk::Image<T,2>::Pointer slice_extraction(typename itk::Image<T,3>::Pointer reader, int index, T)
{
	typedef typename itk::Image<T,3> inImgType;
	typedef typename itk::Image<T,2> outImgType;
	typedef itk::ExtractImageFilter<inImgType,outImgType> FilterType;
	//typedef itk::ImageFileWriter<outImgType> WriterType;

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
	
	inImgType::IndexType start = inputRegion.GetIndex(); 
	start[2]=index;

	inImgType::RegionType desiredRegion; 
	desiredRegion.SetSize(size);
	desiredRegion.SetIndex(start);

	extraction->SetExtractionRegion(desiredRegion);
	extraction->SetInput(reader);

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


	return outImg;
}

/* Explicit instantiations */
template itk::Image<float,2>::Pointer slice_extraction<float> (itk::Image<float,3>::Pointer reader, int index, float);
template itk::Image<unsigned char,2>::Pointer slice_extraction<unsigned char> (itk::Image<unsigned char,3>::Pointer reader, int index, unsigned char);
