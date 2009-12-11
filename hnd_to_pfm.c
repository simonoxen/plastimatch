/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "hnd_io.h"
#include "plm_int.h"
#include "print_and_exit.h"
#include "proj_image.h"

int 
main (int argc, char* argv[]) 
{
    FILE *fp_pfm;
    Proj_image *proj;
    char *hnd_fn, *pfm_fn;

    if (argc != 3) {
	printf ("Usage: hndtopfm hndfile pfmfile\n");
	return 1;
    }

    hnd_fn = argv[1];
    pfm_fn = argv[2];

    /* Read image */
    proj = proj_image_load (hnd_fn, 0);
    if (!proj) {
	print_and_exit ("Couldn't load file for read: %s\n", hnd_fn);
    }

    /* Write image */
    fp_pfm = fopen (pfm_fn, "wb");
    if (!fp_pfm) {
	printf ("Error, cannot open file %s for write\n", pfm_fn);
	exit (1);
    }
    fprintf (fp_pfm, "Pf\n%d %d\n-1\n", proj->dim[0], proj->dim[1]);
    fwrite (proj->img, sizeof(float), proj->dim[0] * proj->dim[1], fp_pfm);
    fclose (fp_pfm);
    return 0;
}
