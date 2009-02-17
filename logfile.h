/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _logfile_h_
#define _logfile_h_

#include <stdio.h>

void logfile_open (FILE** log_fp, char* log_fn);
void logfile_close (FILE** log_fp);
void logfile_printf (FILE* log_fp, char* fmt, ...);

#endif
