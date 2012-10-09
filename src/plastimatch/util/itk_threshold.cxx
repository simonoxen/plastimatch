/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <itkBinaryThresholdImageFilter.h>

#include "itk_threshold.h"

UCharImageType::Pointer
itk_threshold_above (
    FloatImageType::Pointer image, 
    float threshold)
{
    typedef itk::BinaryThresholdImageFilter< 
        FloatImageType, UCharImageType > ThresholdFilterType;
    ThresholdFilterType::Pointer thresh_filter = ThresholdFilterType::New ();
    thresh_filter->SetInput (image);
    thresh_filter->SetLowerThreshold (threshold);
    thresh_filter->SetOutsideValue (0);
    thresh_filter->SetInsideValue (1);
    try {
        thresh_filter->Update ();
    } catch (itk::ExceptionObject & excep) {
        std::cerr << "Exception caught !" << std::endl;
        std::cerr << excep << std::endl;
    }
    return thresh_filter->GetOutput ();
}
