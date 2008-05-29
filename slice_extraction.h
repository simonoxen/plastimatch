
#ifndef _slice_extraction_h
#define _slice_extraction_h


#include "itkImage.h"


/* =======================================================================*
    Definitions
 * =======================================================================*/


typedef float	PixelType;
typedef itk::Image<PixelType, 3>	inImgType;
typedef itk::Image<PixelType, 2>	outImgType;


//outImgType::Pointer slice_extraction(inImgType::Pointer fname);
//void slice_extraction(inImgType::Pointer fname1, outImgType::Pointer fname2);
outImgType::Pointer slice_extraction(inImgType::Pointer fname, int fInd);

#endif
