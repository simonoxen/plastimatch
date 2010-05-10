/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_crop_h_
#define _itk_crop_h_

#include "plm_config.h"

template <class T>
plastimatch1_EXPORT
T
itk_crop (T& image, const int *new_size);

#endif
