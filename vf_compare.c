/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
   Compare two vector fields
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
    printf ("Usage: vf_compare vf1 vf2\n");
    exit (1);
}

void
analyze_volumes (Volume* vol1, Volume* vol2)
{
    int d, i, j, k, v;
    float* img1 = (float*) vol1->img;
    float* img2 = (float*) vol2->img;
    float max_vlen2 = 0.0f;
    int max_vlen_idx_lin = 0;
    int max_vlen_idx[3] = { 0, 0, 0 };

    for (v = 0, k = 0; k < vol1->dim[2]; k++) {
	for (j = 0; j < vol1->dim[1]; j++) {
	    for (i = 0; i < vol1->dim[0]; i++, v++) {
		float* dxyz1 = &img1[3*v];
		float* dxyz2 = &img2[3*v];
		float diff[3];
		float vlen2 = 0.0f;
		for (d = 0; d < 3; d++) {
		    diff[d] = dxyz2[d] - dxyz1[d];
		    vlen2 += diff[d] * diff[d];
		}
		if (vlen2 > max_vlen2) {
		    max_vlen2 = vlen2;
		    max_vlen_idx_lin = v;
		    max_vlen_idx[0] = i;
		    max_vlen_idx[1] = j;
		    max_vlen_idx[2] = k;
		}
	    }
	}
    }

    printf ("Max diff idx:  %4d %4d %4d [%d]\n", max_vlen_idx[0], max_vlen_idx[1], max_vlen_idx[2], max_vlen_idx_lin);
    printf ("Vol 1:         %10.3f %10.3f %10.3f\n", img1[3*max_vlen_idx_lin], img1[3*max_vlen_idx_lin+1], img1[3*max_vlen_idx_lin+2]);
    printf ("Vol 2:         %10.3f %10.3f %10.3f\n", img2[3*max_vlen_idx_lin], img2[3*max_vlen_idx_lin+1], img2[3*max_vlen_idx_lin+2]);
    printf ("Vec len diff:  %10.3f\n", sqrt(max_vlen2));
}

int
main (int argc, char *argv[])
{
    int d;
    char *vf1_fn, *vf2_fn;
    Volume *vol1, *vol2;

    if (argc != 3) {
	print_usage ();
    }
    vf1_fn = argv[1];
    vf2_fn = argv[2];

    vol1 = read_mha (vf1_fn);
    if (!vol1) {
	fprintf (stderr, "Sorry, couldn't open file \"%s\" for read.\n", vf1_fn);
	exit (-1);
    }
    if (vol1->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Sorry, file \"%s\" is not an interleaved float vector field.\n", vf1_fn);
	fprintf (stderr, "Type = %d\n", vol1->pix_type);
	exit (-1);
    }

    vol2 = read_mha (vf2_fn);
    if (!vol2) {
	fprintf (stderr, "Sorry, couldn't open file \"%s\" for read.\n", vf2_fn);
	exit (-1);
    }
    if (vol2->pix_type != PT_VF_FLOAT_INTERLEAVED) {
	fprintf (stderr, "Sorry, file \"%s\" is not an interleaved float vector field.\n", vf2_fn);
	fprintf (stderr, "Type = %d\n", vol2->pix_type);
	exit (-1);
    }

    for (d = 0; d < 3; d++) {
	if (vol1->dim[d] != vol2->dim[d]) {
	    fprintf (stderr, "Can't compare.  Files have different dimensions.\n");
	    exit (-1);
	}
    }

    analyze_volumes (vol1, vol2);

    volume_free (vol1);
    volume_free (vol2);

    return 0;
}
