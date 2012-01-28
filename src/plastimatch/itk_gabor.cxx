/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkGaborImageSource.h"
#include "itkImage.h"

#include "itk_image.h"
#include "itk_crop.h"


void 
itk_gabor (FloatImageType::Pointer image)
{
    typedef itk::GaborImageSource< FloatImageType > GaborSourceType;
    GaborSourceType::Pointer GaborKernelImage = GaborSourceType::New();
}
