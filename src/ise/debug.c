/* -------------------------------------------------------------------------*
    See COPYRIGHT for copyright information.
 * -------------------------------------------------------------------------*/
#include "ise_config.h"
#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int debug_on = 1;
static FILE* gcsfp = 0;

#ifdef _WIN32
static CRITICAL_SECTION debug_cs;
#endif

void
debug_enable (void)
{
    debug_on = 1;
}

void
debug_open (void)
{
    char* filename = "c:\\gcs.txt";
    if (!debug_on) return;
    if (!gcsfp) {
	gcsfp = fopen(filename, "a");
    }
}

void
debug_close (void)
{
    if (!debug_on) return;
    if (gcsfp) {
	fclose(gcsfp);
	gcsfp = 0;
    }
}

void
debug_printf (char* fmt, ...)
{
    static int initialized = 0;
    int was_open = 1;
    va_list argptr;

    if (!debug_on) return;

    va_start (argptr, fmt);
    if (!initialized) {
	initialized = 1;
	InitializeCriticalSection(&debug_cs);
        EnterCriticalSection(&debug_cs);
	if (!gcsfp) {
	    was_open = 0;
	    debug_open();
	}
	fprintf (gcsfp, "=========================\n");
    } else {
        EnterCriticalSection(&debug_cs);
	if (!gcsfp) {
	    was_open = 0;
	    debug_open();
	}
    }

    vfprintf (gcsfp, fmt, argptr);

    va_end (argptr);
    if (!was_open) {
	debug_close ();
    }

    LeaveCriticalSection(&debug_cs);
}

void
mprintf (char* fmt, ...)
{
    char buf[1024];
    va_list argptr;

    va_start (argptr, fmt);
    _vsnprintf (buf, 1024, fmt, argptr);

    MessageBox (NULL, buf, "ISE Warning", MB_OK);

    va_end (argptr);
}
