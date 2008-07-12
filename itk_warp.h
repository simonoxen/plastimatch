/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_warp_h_
#define _itk_warp_h_

#include "itk_image.h"

plastimatch1_EXPORT template<class T, class U> T itk_warp_image (T im_in, DeformationFieldType::Pointer vf, 
				    int linear_interp, U default_val);

#endif
