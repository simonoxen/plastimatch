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
#include "math_util.h"
#include "proton_dose.h"
#include "proton_dose_opts.h"
#include "proj_matrix.h"
#include "readmha.h"
#include "timer.h"

int
main (int argc, char* argv[])
{
    Volume* vol;
    Proton_dose_options options;

    parse_args (&options, argc, argv);

    vol = read_mha (options.input_fn);
    if (!vol) return -1;

    volume_convert_to_float (vol);

    volume_destroy (vol);
    printf ("Done.\n");
    return 0;
}
