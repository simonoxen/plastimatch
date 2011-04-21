/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _mask_image_h_
#define _mask_image_h_

#include "plm_config.h"
#include "plm_image.h"

#if defined (commentout)
template <class T, class U>
T
vector_mask_image (T& vf_image, U& ref_image);
template <class T>
T
vector_mask_image (T& image, float x_spacing,
			float y_spacing, float z_spacing);
template <class T>
T
vector_mask_image (T& vf_image, float* origin, float* spacing, int* size);
template <class T>
T
vector_mask_image (T& vf_image, Plm_image_header* pih);
#endif

template <class T>
T
mask_image (
    T input,
    UCharImageType::Pointer mask,
    int negate_mask,
    float mask_value
);

#endif
