/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <time.h>
#include "plm_version.h"
#include "logfile.h"

#define LOGFILE_ECHO_ON 1

void
logfile_open (FILE** log_fp, char* log_fn)
{
    if (!log_fn[0]) return;
    if (!(*log_fp)) {
	*log_fp = fopen (log_fn, "w");
	if (!*log_fp) {
	    /* If failure (e.g. bad path), do nothing */	
	}
    } else {
	/* Already open? */
    }
    logfile_printf (*log_fp, "Plastimatch " 
		     PLASTIMATCH_VERSION_STRING
		     "\n");
}

void
logfile_close (FILE** log_fp)
{
    if (*log_fp) {
	fclose (*log_fp);
	*log_fp = 0;
    }
}

void
logfile_printf (FILE* log_fp, char* fmt, ...)
{
    va_list argptr;
    va_start (argptr, fmt);

    /* Write to console */
    if (LOGFILE_ECHO_ON) {
	vprintf (fmt, argptr);
	fflush (stdout);
    }

    if (!log_fp) {
	va_end (argptr);
	return;
    }

    /* Write to file */
    vfprintf (log_fp, fmt, argptr);
    fflush (log_fp);

    va_end (argptr);
}
