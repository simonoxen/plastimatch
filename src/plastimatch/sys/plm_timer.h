/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_timer_h_
#define _plm_timer_h_

#include "plmsys_config.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

class Plm_timer {
public:
    double start_time;
#ifdef _WIN32
    LARGE_INTEGER clock_freq;
#endif
};

#endif
