/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "plm_config.h"
#include "demons.h"

/* GCS FIX: ITK uses sum_d(pix_spacing[d]^2) / (#dim) for homog */
gpuit_EXPORT
void
demons_default_parms (DEMONS_Parms* parms)
{
    parms->accel = 1.0;
    parms->denominator_eps = 1.0;
    parms->filter_width[0] = 3;
    parms->filter_width[1] = 3;
    parms->filter_width[2] = 3;
    parms->homog = 1.0;
    parms->max_its = 10;
    parms->filter_std = 5.0;
}

gpuit_EXPORT
Volume*
demons (Volume* fixed, Volume* moving, Volume* moving_grad, Volume* vf_init, char* method, DEMONS_Parms* parms)
{
#if HAVE_BROOK
    if (!strcmp (method, "BROOK") || !strcmp (method, "brook")) {
	return demons_brook (fixed, moving, moving_grad, vf_init, parms);
    }
#endif
    return demons_c (fixed, moving, moving_grad, vf_init, parms);
}
