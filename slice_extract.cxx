/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkExtractImageFilter.h"
#include "itkImage.h"
#include "itk_image.h"
#include "slice_extract.h"

template<class T>
typename itk::Image<T,2>::Pointer slice_extract 
(
    typename itk::Image<T,3>::Pointer reader, 
    int slice_no
)
{
    typedef typename itk::Image<T,3> inImgType;
    typedef typename itk::Image<T,2> outImgType;
    typedef typename itk::ExtractImageFilter<inImgType,outImgType> FilterType;

    typename FilterType::Pointer extraction=FilterType::New();
	
    try {
	reader->Update(); 
	//std::cout << "Ho letto!" << std::endl;
    }
    catch ( itk::ExceptionObject &err) {
	std::cout << "ExceptionObject caught a !" << std::endl; 
	std::cout << err << std::endl; 
	//return -1;
    }
	
    typename inImgType::RegionType inputRegion=reader->GetLargestPossibleRegion();
    typename inImgType::SizeType size = inputRegion.GetSize();
    size[2] = 0;
	
    typename inImgType::IndexType start = inputRegion.GetIndex(); 
    start[2]=slice_no;

    typename inImgType::RegionType desiredRegion; 
    desiredRegion.SetSize(size);
    desiredRegion.SetIndex(start);

    extraction->SetExtractionRegion(desiredRegion);
    extraction->SetInput(reader);

    typename outImgType::Pointer outImg = outImgType::New();
	
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
/* RMK: Visual studio 2005 without service pack requires <float> specifier
   on the explicit extantiations.  The current hypothesis is that this 
   is because the template is nested. */
template plastimatch1_EXPORT itk::Image<unsigned char,2>::Pointer slice_extract<unsigned char> (itk::Image<unsigned char,3>::Pointer reader, int index);
template plastimatch1_EXPORT itk::Image<unsigned int,2>::Pointer slice_extract<unsigned int> (itk::Image<unsigned int,3>::Pointer reader, int index);
template plastimatch1_EXPORT itk::Image<unsigned long,2>::Pointer slice_extract<unsigned long> (itk::Image<unsigned long,3>::Pointer reader, int index);
template plastimatch1_EXPORT itk::Image<float,2>::Pointer slice_extract<float> (itk::Image<float,3>::Pointer reader, int index);
#if defined (commentout)
template plastimatch1_EXPORT UCharVecImage2DType::Pointer slice_extract<UCharVecType> (UCharVecImageType::Pointer reader, int index);
#endif
