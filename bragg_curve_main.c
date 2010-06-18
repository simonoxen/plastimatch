/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif
#include "bragg_curve_opts.h"

int
main (int argc, char* argv[])
{
    Bragg_curve_options options;

    parse_args (&options, argc, argv);

    printf ("Done.\n");
    return 0;
}
