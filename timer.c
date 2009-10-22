/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "timer.h"

Timer *
plm_timer_create (void)
{
    Timer *timer;
    int rc;

    timer = (Timer*) malloc (sizeof(Timer));
    if (!timer) return 0;
    
    rc = gettimeofday (&timer->tv, 0);

    return timer;
}

void
plm_timer_start (Timer *timer)
{
    int rc;

    rc = gettimeofday (&timer->tv, 0);
}

double
plm_timer_report (Timer *timer)
{
    struct timeval tv;
    double interval;
    int rc;
    
    rc = gettimeofday (&tv, 0);

    interval = (double) (tv.tv_sec - timer->tv.tv_sec);
    interval += ((double) (tv.tv_usec - timer->tv.tv_usec)) / 1000000.;

    return interval;
}


void
plm_timer_destroy (Timer *timer)
{
    free (timer);
}
