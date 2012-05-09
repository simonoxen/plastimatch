/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_adjust_h_
#define _itk_adjust_h_

#include "plmutil_config.h"
#include <list>
#include <utility>
#include "itk_image_type.h"

typedef std::list< std::pair< float, float > > Adjustment_list;

/* Does destructive, in-place adjustment */
PLMUTIL_API void itk_adjust (FloatImageType::Pointer image, const Adjustment_list& al);
PLMUTIL_API void itk_auto_adjust (FloatImageType::Pointer image);

#endif
