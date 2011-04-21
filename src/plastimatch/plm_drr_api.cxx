/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define PLM_INTERNAL
#include "plm_drr_api.h"

Plm_drr_context* 
plm_drr_context_create ()
{
    Plm_drr_context* prc = (Plm_drr_context*) calloc (
	1, sizeof (Plm_drr_context));

    return prc;
}

void
plm_drr_context_destroy (Plm_drr_context* prc)
{
    free (prc->command_string);
    free (prc);
}
