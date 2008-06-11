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
#include "volume.h"
#include "readmha.h"

void
print_usage (void)
{
    printf ("Usage: vf_stats vf_file\n");
    exit (1);
}

void
analyze_volume (Volume* vol)
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

int
main (int argc, char *argv[])
{
    char* vf_fn;
    Volume* vol;

    if (argc != 2) {
	print_usage ();
    }
    vf_fn = argv[1];

    vol = read_mha (vf_fn);
    if (!vol) {
	fprintf (stderr, "Sorry, couldn't open file \"%s\" for read.\n", vf_fn);
	exit (-1);
    }

    if (vol->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Sorry, file \"%s\" is not an interleaved float vector field.\n", vf_fn);
	fprintf (stderr, "Type = %d\n");
	exit (-1);
    }

    analyze_volume (vol);

    volume_free (vol);

    return 0;
}
