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

#include "plmbase.h"

#include "plm_math.h"
#include "proton_dose.h"

int
main (int argc, char* argv[])
{
    Volume *ct, *dose;
    Proton_dose_parms parms;

    parms.parse_args (argc, argv);

    ct = read_mha (parms.input_fn);
    if (!ct) return -1;

    volume_convert_to_float (ct);

    printf ("Working... ");
    fflush(stdout);

    dose = volume_clone_empty (ct);

    proton_dose_compute (dose, ct, &parms);

    write_mha (parms.output_fn, dose);

    delete ct;
    delete dose;
    printf ("done.  \n\n");
    return 0;
}
