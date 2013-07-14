/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "plm_math.h"
#include "ion_dose.h"
#include "ion_parms.h"
#include "ion_plan.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Ion_parms parms;

    Ion_plan::Pointer scene = parms.get_scene ();
    if (!parms.parse_args (argc, argv)) {
        exit (0);
    }

    printf ("Working...\n");
    fflush(stdout);

    scene->compute_dose ();
    Plm_image::Pointer dose = scene->get_dose ();
    dose->save_image (parms.output_dose_fn.c_str());
    printf ("done.  \n\n");

    return 0;
}
