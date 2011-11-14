/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_directions_h_
#define _itk_directions_h_

#include "plm_config.h"
#include "direction_cosines.h"
#include "itk_image.h"

void
itk_direction_from_dc (DirectionType* itk_dc, const Direction_cosines& dc)
{
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    (*itk_dc)[d1][d2] = dc[d1*3+d2];
	}
    }
}

#endif
