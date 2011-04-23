/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bspline.h"
#include "bspline_opts.h"
#include "mha_io.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Bspline_xform *bxf;

    if (argc != 3) {
	printf ("Error, invalid arguments\n");
	return -1;
    }

    bxf = bspline_xform_load (argv[1]);
    if (!bxf) {
	return -1;
    }

    bspline_xform_save (bxf, argv[2]);

    bspline_xform_free (bxf);
    free (bxf);

    return 0;
}
