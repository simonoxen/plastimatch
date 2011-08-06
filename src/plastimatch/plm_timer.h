/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_timer_h_
#define _plm_timer_h_

#include "plm_config.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class plmsys_EXPORT Plm_timer {
public:
    double start_time;
#ifdef _WIN32
    LARGE_INTEGER clock_freq;
#endif
};

plmsys_EXPORT
void plm_timer_start (Plm_timer *timer);
plmsys_EXPORT
double plm_timer_report (Plm_timer *timer);

#endif
