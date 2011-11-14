/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_directions_h_
#define _itk_directions_h_

#include "plm_config.h"
#include "direction_cosines.h"
#include "itk_image.h"

void
itk_direction_from_dc (DirectionType* itk_dc, const Direction_cosines& dc);

#endif
