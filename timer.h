/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _timer_h_
#define _timer_h_

#include "plm_config.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

typedef struct timer_struct Timer;
struct timer_struct {
    double start_time;
#ifdef _WIN32
    LARGE_INTEGER pc_freq;
#endif
};

#if defined __cplusplus
extern "C" {
#endif
Timer *plm_timer_create (void);
gpuit_EXPORT
void plm_timer_start (Timer *timer);
gpuit_EXPORT
double plm_timer_report (Timer *timer);
gpuit_EXPORT
void plm_timer_destroy (Timer *timer);
#if defined __cplusplus
}
#endif

#endif
