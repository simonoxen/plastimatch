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
    Plm_image* ct;
    Volume* dose;
    Proton_Parms parms;

    parms.parse_args (argc, argv);

    // TODO: Move this into Proton_Parms::parse_args()
    /* ----------------------------- */
    ct = plm_image_load (parms.input_fn, PLM_IMG_TYPE_ITK_FLOAT);
    if (!ct) return -1;

    parms.scene->set_patient (ct->gpuit_float ());

    /* try to setup the scene with the provided parameters */
    if (!parms.scene->init (parms.ray_step)) {
        fprintf (stderr, "ERROR: Unable to initilize scene.\n");
        exit (0);
    }
    /* ----------------------------- */

    printf ("Working...\n");
    fflush(stdout);

    dose = proton_dose_compute (&parms);

    write_mha (parms.output_fn, dose);

    delete ct;
    delete dose;
    printf ("done.  \n\n");
    return 0;
}
