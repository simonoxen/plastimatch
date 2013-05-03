/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mha_io.h"
#include "plm_math.h"
#include "proton_dose.h"
#include "proton_parms.h"
#include "proton_scene.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Volume* dose;
    Proton_parms parms;
    //Proton_scene scene;
    //parms.set_scene (&scene);

    Proton_scene::Pointer scene = parms.get_scene ();
    if (!parms.parse_args (argc, argv)) {
        exit (0);
    }

    printf ("Working...\n");
    fflush(stdout);

    dose = proton_dose_compute (scene);
    write_mha (parms.output_dose_fn.c_str(), dose);
    printf ("done.  \n\n");

    delete dose;
    return 0;
}
