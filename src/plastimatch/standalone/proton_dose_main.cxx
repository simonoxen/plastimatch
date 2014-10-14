/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#include "RTP_dose.h"
#include "RTP_parms.h"
#include "RTP_plan.h"
#include "plm_math.h"
#include "plm_timer.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Plm_timer timer;
    RTP_parms parms;
    timer.start ();
    if (!parms.parse_args (argc, argv)) {
        return 1;
    }

    printf("Execution time : %f secondes.\n", timer.report ());
    return 0;
}
