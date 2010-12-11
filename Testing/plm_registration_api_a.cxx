/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "plm_registration_api.h"

int
main (int argc, char* argv[])
{
    Plm_registration_context* prc;

    prc = plm_registration_context_create ();

    plm_registration_context_destroy (prc);
}
