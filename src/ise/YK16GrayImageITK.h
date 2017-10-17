#include "itkImage.h"
typedef itk::Image<unsigned short, 2> UnsignedShortImageType;

void CopyYKImage2ItkImage(YK16GrayImage* pYKImage, UnsignedShortImageType::Pointer& spTarImage);
void CopyItkImage2YKImage(UnsignedShortImageType::Pointer& spSrcImage, YK16GrayImage* pYKImage);
    
