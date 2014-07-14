/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#include "ion_dose.h"
#include "ion_parms.h"
#include "ion_plan.h"
#include "plm_math.h"
#include "plm_timer.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Plm_timer timer;
    Ion_parms parms;
    timer.start ();
    if (!parms.parse_args (argc, argv)) {
        return 1;
    }

    printf("Execution time : %f secondes.\n", timer.report ());
    return 0;
}
