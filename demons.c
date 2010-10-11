/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "demons.h"
#include "threading.h"

void
demons_default_parms (DEMONS_Parms* parms)
{
    parms->threading = THREADING_CPU_OPENMP;
    parms->accel = 1.0;
    parms->denominator_eps = 1.0;
    parms->filter_width[0] = 3;
    parms->filter_width[1] = 3;
    parms->filter_width[2] = 3;
    /* GCS FIX: ITK uses sum_d(pix_spacing[d]^2) / (#dim) for homog */
    parms->homog = 1.0;
    parms->max_its = 10;
    parms->filter_std = 5.0;
}

Volume*
demons (
    Volume* fixed, 
    Volume* moving, 
    Volume* moving_grad, 
    Volume* vf_init, 
    DEMONS_Parms* parms
)
{
    switch (parms->threading) {
#if BROOK_FOUND
    case THREADING_BROOK:
	return demons_brook (fixed, moving, moving_grad, vf_init, parms);
#endif
#if CUDA_FOUND
    case THREADING_CUDA:
	return demons_cuda (fixed, moving, moving_grad, vf_init, parms);
#endif
#if OPENCL_FOUND
    case THREADING_OPENCL:
	return demons_opencl (fixed, moving, moving_grad, vf_init, parms);
#endif
    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
    default:
	return demons_c (fixed, moving, moving_grad, vf_init, parms);
    }
}
