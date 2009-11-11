/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if HAVE_SYS_STAT
#include <sys/stat.h>
#endif
#if _WIN32
#include <direct.h>
#endif

#if (_WIN32)
#define ISSLASH(c) (((c) == '/') || ((c) == '\\'))
#else
#define ISSLASH(c) ((c) == '/')
#endif


int
file_exists (const char *filename)
{
    FILE *f = fopen (filename, "r");
    if (f) {
	fclose (f);
	return 1;
    }
    return 0;
}

void
make_directory (const char *dirname)
{
#if (_WIN32)
    mkdir (dirname);
#else
    mkdir (dirname, 0777);
#endif
}

void
make_directory_recursive (const char *dirname)
{
    char *p, *tmp;

    if (!dirname) return;

    tmp = strdup (dirname);
    p = tmp;
    while (*p) {
	if (ISSLASH (*p)) {
	    *p = 0;
	    make_directory (tmp);
	    *p = '/';
	}
	p++;
    }
    free (tmp);
}
