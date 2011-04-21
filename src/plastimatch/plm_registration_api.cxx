/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define PLM_INTERNAL
#include "plm_registration_api.h"

Plm_registration_context* 
plm_registration_context_create ()
{
    Plm_registration_context* prc = (Plm_registration_context*) calloc (
	1, sizeof (Plm_registration_context));

    return prc;
}

void
plm_registration_context_destroy (Plm_registration_context* prc)
{
    free (prc->command_string);
    free (prc);
}
