/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_trilin_h_
#define _drr_trilin_h_

#include "plm_config.h"
#include "plmbase.h"

gpuit_EXPORT
double
drr_trace_ray_trilin_approx (Volume* vol, double* p1in, double* p2in);
gpuit_EXPORT
double
drr_trace_ray_trilin_exact (Volume* vol, double* p1in, double* p2in);

#endif
