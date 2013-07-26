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
    if (!parms.parse_args (argc, argv)) {
        return 1;
    }
    return 0;
}
