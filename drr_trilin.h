/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_trilin_h_
#define _drr_trilin_h_

#include "volume.h"

double
drr_trace_ray_trilin_approx (Volume* vol, double* p1in, double* p2in);
double
drr_trace_ray_trilin_exact (Volume* vol, double* p1in, double* p2in);

#endif
