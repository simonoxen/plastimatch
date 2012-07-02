/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_threshold_h_
#define _itk_threshold_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

PLMUTIL_API 
UCharImageType::Pointer itk_threshold_above (FloatImageType::Pointer image, 
    float threshold);

#endif
