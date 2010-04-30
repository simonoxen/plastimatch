/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include "volume.h"
#include "volume_limit.h"

#define DRR_BOUNDARY_TOLERANCE 1e-6

void
volume_limit_set (Volume_limit *vol_limit, Volume *vol)
{
    int d;

    /* Compute volume boundary box */
    for (d = 0; d < 3; d++) {
	vol_limit->limits[d][0] = vol->offset[d] - 0.5 * vol->pix_spacing[d];
	vol_limit->limits[d][1] = vol_limit->limits[d][0] 
	    + vol->dim[d] * vol->pix_spacing[d];
	if (vol_limit->limits[d][0] <= vol_limit->limits[d][1]) {
	    vol_limit->dir[d] = 0;
	    vol_limit->limits[d][0] += DRR_BOUNDARY_TOLERANCE;
	    vol_limit->limits[d][1] -= DRR_BOUNDARY_TOLERANCE;
	} else {
	    vol_limit->dir[d] = 1;
	    vol_limit->limits[d][0] -= DRR_BOUNDARY_TOLERANCE;
	    vol_limit->limits[d][1] += DRR_BOUNDARY_TOLERANCE;
	}
    }
}
