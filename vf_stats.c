/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Analyze a vector field for invertibility, smoothness.
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "readmha.h"
#include "vf_stats.h"
#include "volume.h"

void
vf_analyze (Volume* vol)
{
    int d, i, j, k, v;
    float* img = (float*) vol->img;
    float mean_av[3], mean_v[3];
    float mins[3];
    float maxs[3];

    for (d = 0; d < 3; d++) {
	mean_av[d] = mean_v[d] = 0.0;
	mins[d] = maxs[d] = img[d];
    }

    for (v = 0, k = 0; k < vol->dim[2]; k++) {
	for (j = 0; j < vol->dim[1]; j++) {
	    for (i = 0; i < vol->dim[0]; i++, v++) {
		float* dxyz = &img[3*v];
		for (d = 0; d < 3; d++) {
		    mean_v[d] += dxyz[d];
		    mean_av[d] += fabs(dxyz[d]);
		    if (dxyz[d] > maxs[d]) {
			maxs[d] = dxyz[d];
		    } else if (dxyz[d] < mins[d]) {
			mins[d] = dxyz[d];
		    }
		}
	    }
	}
    }
    for (d = 0; d < 3; d++) {
	mean_v[d] /= vol->npix;
	mean_av[d] /= vol->npix;
    }

    printf ("Min:       %10.3f %10.3f %10.3f\n", mins[0], mins[1], mins[2]);
    printf ("Mean:      %10.3f %10.3f %10.3f\n", mean_v[0], mean_v[1], mean_v[2]);
    printf ("Max:       %10.3f %10.3f %10.3f\n", maxs[0], maxs[1], maxs[2]);
    printf ("Mean abs:  %10.3f %10.3f %10.3f\n", mean_av[0], mean_av[1], mean_av[2]);
}
