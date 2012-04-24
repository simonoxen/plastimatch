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

#include "libplmimage.h"

#include "math_util.h"
#include "plm_timer.h"
#include "proton_dose.h"
#include "proton_dose_opts.h"
#include "proj_matrix.h"

int
main (int argc, char* argv[])
{
    Volume *ct, *dose;
    Proton_dose_options options;

    proton_dose_parse_args (&options, argc, argv);

    ct = read_mha (options.input_fn);
    if (!ct) return -1;

    volume_convert_to_float (ct);

    printf ("Working... ");
    fflush(stdout);

    dose = volume_clone_empty (ct);

    proton_dose_compute (dose, ct, &options);

    write_mha (options.output_fn, dose);

    delete ct;
    delete dose;
    printf ("done.  \n\n");
    return 0;
}
