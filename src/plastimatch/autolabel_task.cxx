/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImageRegionIterator.h"

#include "autolabel_task.h"
#include "dlib_trainer.h"

Dlib_trainer::Dense_sample_type 
Autolabel_task::make_sample (const FloatImageType::Pointer& thumb_img)
{
    itk::ImageRegionIterator< FloatImageType > thumb_it (
        thumb_img, thumb_img->GetLargestPossibleRegion());
    Dlib_trainer::Dense_sample_type d;
    for (int j = 0; j < 256; j++) {
        d(j) = thumb_it.Get();
        ++thumb_it;
    }
    return d;
}

