/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkGaborImageSource.h"
#include "itkGaborKernelFunction.h"
#include "itkImage.h"

#include "itk_gabor.h"
#include "itk_image.h"
#include "itk_image_save.h"


void 
itk_gabor (FloatImageType::Pointer image)
{
    typedef itk::GaborImageSource< FloatImageType > GaborSourceType;
    GaborSourceType::Pointer GaborKernelImage = GaborSourceType::New();

    GaborKernelImage->Update();
    FloatImageType::Pointer img = GaborKernelImage->GetOutput();

    itk_image_save (img, "tmp.mha");
}
