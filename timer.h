/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _timer_h_
#define _timer_h_

#include "plm_config.h"
#include <sys/time.h>

typedef struct timer_struct Timer;
struct timer_struct {

    struct timeval tv;
    
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
