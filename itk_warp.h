/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_warp_h_
#define _itk_warp_h_

#include "itk_image.h"

template<class T> T itk_warp_image (T im_in, T im_sz, DeformationFieldType::Pointer vf, 
				    int linear_interp, float default_val);

#endif
