/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "plm_int.h"
#include "itkImage.h"
#include "itk_image.h"
#include "itkAndConstantToImageFilter.h"

UCharImageType::Pointer
ss_img_extract (UInt32ImageType::Pointer image, int bit)
{
    typedef itk::AndConstantToImageFilter< UInt32ImageType, 
	uint32_t, UCharImageType > AndFilterType;

    AndFilterType::Pointer and_filter = AndFilterType::New();

    and_filter->SetInput (image);
    and_filter->SetConstant (1 << bit);
    try {
	and_filter->Update ();
    }
    catch (itk::ExceptionObject &err) {
	std::cout << "Exception during and operation." << std::endl;
	std::cout << err << std::endl;
	exit (1);
    }
    return and_filter->GetOutput ();
}
