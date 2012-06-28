/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _logfile_h_
#define _logfile_h_

/*
 *  You probably do not want to #include this header directly.
 *
 *   Instead, it is preferred to #include "plmsys.h"
 */

#include "plmsys_config.h"

PLMSYS_C_API void logfile_open (char* log_fn);
PLMSYS_C_API void logfile_close (void);
PLMSYS_C_API void logfile_printf (const char* fmt, ...);
#define lprintf logfile_printf

#endif