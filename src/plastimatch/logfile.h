/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _logfile_h_
#define _logfile_h_

#include "plm_config.h"
#include <stdio.h>

#define lprintf logfile_printf

#if defined __cplusplus
extern "C" {
#endif
plmsys_EXPORT
void logfile_open (char* log_fn);
plmsys_EXPORT
void logfile_close (void);
plmsys_EXPORT
void logfile_printf (const char* fmt, ...);
#if defined __cplusplus
}
#endif

#endif
