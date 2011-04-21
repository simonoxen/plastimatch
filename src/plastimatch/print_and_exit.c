/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "print_and_exit.h"

#if (defined(_WIN32) || defined(WIN32))
	#define vsnprintf _vsnprintf
#endif

void
print_and_wait (char* prompt_fmt, ...)
{
    if (prompt_fmt) {
        va_list argptr;
	va_start (argptr, prompt_fmt);
	vprintf (prompt_fmt, argptr);
	va_end (argptr);
    }
    printf ("Hit any key to continue\n");
    getchar();
}

void
print_and_exit (char* prompt_fmt, ...)
{
    if (prompt_fmt) {
        va_list argptr;
	va_start (argptr, prompt_fmt);
	vprintf (prompt_fmt, argptr);
	va_end (argptr);
    }
    exit (1);
}
