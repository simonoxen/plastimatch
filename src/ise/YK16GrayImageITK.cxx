#include "YK16GrayImage.h"
#include "itkImageRegionIterator.h"

void
YK16GrayImage::CopyYKImage2ItkImage (
    YK16GrayImage* pYKImage, UnsignedShortImageType::Pointer& spTarImage)
{
    if (pYKImage == NULL)
        return;
    //Raw File open	
    //UnsignedShortImageType::SizeType tmpSize = 
    UnsignedShortImageType::RegionType region = spTarImage->GetRequestedRegion();
    UnsignedShortImageType::SizeType tmpSize = region.GetSize();

    int sizeX = tmpSize[0];
    int sizeY = tmpSize[1];

    if (sizeX < 1 || sizeY <1)
        return;

    itk::ImageRegionIterator<UnsignedShortImageType> it(spTarImage, region);

    int i = 0;
    for (it.GoToBegin() ; !it.IsAtEnd(); ++it)
    {
        it.Set(pYKImage->m_pData[i]);
        i++;
    }
    //int totCnt = i;
    //writerType::Pointer writer = writerType::New();	
    //writer->SetInput(spTarImage);	
    //writer->SetFileName("C:\\ThisImageIs_spSrcImage.png");	//It works!
    //writer->Update();
}

void
YK16GrayImage::CopyItkImage2YKImage (
    UnsignedShortImageType::Pointer& spSrcImage, YK16GrayImage* pYKImage)
{
	if (pYKImage == NULL)
		return;
	//Raw File open	
	//UnsignedShortImageType::SizeType tmpSize = 
	UnsignedShortImageType::RegionType region = spSrcImage->GetRequestedRegion();
	UnsignedShortImageType::SizeType tmpSize = region.GetSize();

	int sizeX = tmpSize[0];
	int sizeY = tmpSize[1];

	if (sizeX < 1 || sizeY <1)
		return;

	//itk::ImageRegionConstIterator<UnsignedShortImageType> it(spSrcImage, region);
	itk::ImageRegionIterator<UnsignedShortImageType> it(spSrcImage, region);

	int i = 0;
	for (it.GoToBegin() ; !it.IsAtEnd() ; ++it)
	{
		pYKImage->m_pData[i] = it.Get();
		i++;
	}
	//int totCnt = i; //Total Count is OK

	//int width = pYKImage->m_iWidth;
	//int height = pYKImage->m_iHeight;


	//writerType::Pointer writer = writerType::New();	
	//writer->SetInput(spSrcImage);	
	//writer->SetFileName("C:\\ThisImageIs_spSrcImage2.png");	//It works!
	//writer->Update();
}
