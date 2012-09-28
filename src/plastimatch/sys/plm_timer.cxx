/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"
#include <stdlib.h>
#include <stdio.h>

#include "plm_timer_p.h"
#include "plm_timer.h"

#include "compiler_warnings.h"

double
Plm_timer_private::get_time ()
{
#if defined (_WIN32)
    LARGE_INTEGER clock_count;
    QueryPerformanceCounter (&clock_count);
    return ((double) (clock_count.QuadPart)) / ((double) this->clock_freq.QuadPart);
#else
    struct timeval tv;
    int rc;
    rc = gettimeofday (&tv, 0);
    UNUSED_VARIABLE (rc);
    return ((double) tv.tv_sec) + ((double) tv.tv_usec) / 1000000.;
#endif
}

Plm_timer::Plm_timer ()
{
    this->d_ptr = new Plm_timer_private;
}

Plm_timer::~Plm_timer ()
{
    delete this->d_ptr;
}

void
Plm_timer::start ()
{
#if defined (_WIN32)
    QueryPerformanceFrequency (&d_ptr->clock_freq);
#endif
    d_ptr->start_time = d_ptr->get_time ();
}

double
Plm_timer::report ()
{
    double current_time;

    current_time = d_ptr->get_time ();
    return current_time - d_ptr->start_time;
}
