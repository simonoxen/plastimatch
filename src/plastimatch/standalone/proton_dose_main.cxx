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

int
main (int argc, char* argv[])
{
    Volume *ct, *dose;
    Proton_Parms parms;

    parms.parse_args (argc, argv);

    // TODO: Move this into Proton_Parms::parse_args()
    //       and move away from read_mha in favor of plm_image
    /* ----------------------------- */
    ct = read_mha (parms.input_fn);
    if (!ct) return -1;

    volume_convert_to_float (ct);
    parms.scene->set_patient (ct);

    /* try to setup the scene with the provided parameters */
    if (!parms.scene->init (parms.ray_step)) {
        fprintf (stderr, "ERROR: Unable to initilize scene.\n");
        exit (0);
    }
    /* ----------------------------- */

    printf ("Working... ");
    fflush(stdout);

    dose = proton_dose_compute (&parms);

    write_mha (parms.output_fn, dose);

    delete ct;
    delete dose;
    printf ("done.  \n\n");
    return 0;
}
