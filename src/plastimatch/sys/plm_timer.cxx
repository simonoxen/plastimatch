/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"
#include <stdlib.h>
#include <stdio.h>

#include "plmsys.h"

#include "compiler_warnings.h"
#include "plm_timer.h"

static double
plm_timer_get_time (Plm_timer *timer)
{
#if defined (_WIN32)
    LARGE_INTEGER clock_count;
    QueryPerformanceCounter (&clock_count);
    return ((double) (clock_count.QuadPart)) / ((double) timer->clock_freq.QuadPart);
#else
    struct timeval tv;
    int rc;
    rc = gettimeofday (&tv, 0);
    UNUSED_VARIABLE (rc);
    return ((double) tv.tv_sec) + ((double) tv.tv_usec) / 1000000.;
#endif
}

Plm_timer*
plm_timer_create ()
{
    return new Plm_timer;
}

void
plm_timer_destroy (Plm_timer *timer)
{
    delete timer;
}

void
plm_timer_start (Plm_timer *timer)
{
#if defined (_WIN32)
    QueryPerformanceFrequency (&timer->clock_freq);
#endif
    timer->start_time = plm_timer_get_time (timer);
}

double
plm_timer_report (Plm_timer *timer)
{
    double current_time;

    current_time = plm_timer_get_time (timer);
    return current_time - timer->start_time;
}