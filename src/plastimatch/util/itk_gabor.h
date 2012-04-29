/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_gabor_h_
#define _itk_gabor_h_

#include "plm_config.h"
#include "itk_image.h"

plastimatch1_EXPORT 
void itk_gabor (FloatImageType::Pointer image);
plastimatch1_EXPORT 
FloatImageType::Pointer
itk_gabor_create (const Plm_image_header *pih);

#endif
