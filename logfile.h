/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _logfile_h_
#define _logfile_h_

#include <stdio.h>

#if defined __cplusplus
extern "C" {
#endif
gpuit_EXPORT
void logfile_open (char* log_fn);
gpuit_EXPORT
void logfile_close (void);
gpuit_EXPORT
void logfile_printf (char* fmt, ...);
#if defined __cplusplus
}
#endif

#endif
