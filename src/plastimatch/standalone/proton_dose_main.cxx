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

#include "mha_io.h"
#include "plm_math.h"
#include "proton_dose.h"
#include "proton_parms.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Volume* dose;
    Proton_Parms parms;

    if (!parms.parse_args (argc, argv)) {
        exit (0);
    }

    printf ("Working...\n");
    fflush(stdout);

    dose = proton_dose_compute (&parms);
    write_mha (parms.output_fn, dose);
    printf ("done.  \n\n");

    delete dose;
    return 0;
}
