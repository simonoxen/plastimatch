/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itk_directions.h"

void
itk_direction_from_dc (DirectionType* itk_dc, const Direction_cosines& dc)
{
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    (*itk_dc)[d1][d2] = dc[d1*3+d2];
	}
    }
}
