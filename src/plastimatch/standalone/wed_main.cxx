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
#include "plmdose.h"

#include "plm_math.h"

void
wed_ct_compute (
    char* out_fn,
    Wed_Parms* parms
)
{
    rpl_volume_save (parms->scene->rpl_vol, out_fn);
}

int
main (int argc, char* argv[])
{
    Wed_Parms parms;

    if (!parms.parse_args (argc, argv)) {
        exit (0);
    }

    printf ("Working...\n");
    fflush(stdout);

    wed_ct_compute ("foo.mha", &parms);

    printf ("done.  \n\n");

    return 0;
}
