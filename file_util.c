/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif
#if HAVE_SYS_STAT
#include <sys/stat.h>
#endif

#if (_WIN32)
#define ISSLASH(c) (((c) == '/') || ((c) == '\\'))
#else
#define ISSLASH(c) ((c) == '/')
#endif

int
is_directory (const char *dir)
{
#if (defined(_WIN32) || defined(WIN32))
    char pwd[_MAX_PATH];
    if (!_getcwd (pwd, _MAX_PATH)) {
        return 0;
    }
    if (_chdir (dir) == -1) {
        return 0;
    }
    _chdir (pwd);
#else /* UNIX */
    DIR *dp;
    if ((dp = opendir (dir)) == NULL) {
        return 0;
    }
    closedir (dp);
#endif
    return 1;
}

int
extension_is (char* fname, char* ext)
{
    return (strlen (fname) > strlen(ext)) 
	&& !strcmp (&fname[strlen(fname)-strlen(ext)], ext);
}

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
