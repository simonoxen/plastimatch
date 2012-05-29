/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "plmbase.h"
#include "plmsegment.h"

#include "plm_math.h"

int
main (int argc, char* argv[])
{
    Mabs_parms parms;

    if (!parms.parse_args (argc, argv)) {
        exit (0);
    }

    printf ("Initializing MABS...\n");

    if (parms.debug) {
        parms.print ();
    }

    return 0;
}
