/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#include "plm_math.h"
#include "plm_timer.h"
#include "rt_dose.h"
#include "rt_parms.h"
#include "rt_plan.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Plm_timer timer;
    Rt_parms parms;
    timer.start ();
    if (!parms.parse_args (argc, argv)) {
        return 1;
    }

    printf("Execution time : %f secondes.\n", timer.report ());
    return 0;
}
