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
    BSPLINE_Xform *bxf;

    if (argc != 3) {
	printf ("Error, invalid arguments\n");
	return -1;
    }

    bxf = read_bxf (argv[1]);
    if (!bxf) {
	return -1;
    }

    write_bxf (argv[2], bxf);

    bspline_xform_free (bxf);
    free (bxf);

    return 0;
}
