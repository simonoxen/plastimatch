/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "plm_math.h"
#include "proton_dose.h"
#include "proton_parms.h"
#include "proton_scene.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Proton_parms parms;

    Proton_scene::Pointer scene = parms.get_scene ();
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
