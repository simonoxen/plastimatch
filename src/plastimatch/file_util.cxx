/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <windows.h>
#include <direct.h>
#include <io.h>
#else
#include <unistd.h>    /* sleep() */
#include <dirent.h>
#endif
#if HAVE_SYS_STAT
#include <sys/stat.h>
#endif
#include <sys/stat.h>

#include "bstring_util.h"
#include "file_util.h"

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
extension_is (const char* fname, const char* ext)
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

uint64_t
file_size (const char *filename)
{
    struct stat fs;
    if (stat (filename, &fs) != 0) return 0;

    return (uint64_t) fs.st_size;
}

void
make_directory (const char *dirname)
{
    int retries = 4;

#if (_WIN32)
    mkdir (dirname);
#else
    mkdir (dirname, 0777);
#endif

    /* On various samba mounts, there is a delay in creating the directory. 
       Here, we work around that problem by waiting until the directory 
       is created */
    while (--retries > 0 && !is_directory (dirname)) {
#if (_WIN32)
	Sleep (1000);
#else
	sleep (1);
#endif
    }
}

void
make_directory_recursive (const char *dirname)
{
    char *p, *tmp;

    if (!dirname) return;

    tmp = strdup (dirname);
    p = tmp;
    while (*p) {
	if (ISSLASH (*p) && p != tmp) {
	    *p = 0;
	    make_directory (tmp);
	    *p = '/';
	}
	p++;
    }
    free (tmp);
}

FILE*
make_tempfile (void)
{
# if defined (_WIN32)
    /* tmpfile is broken on windows.  It tries to create the 
	temorary files in the root directory where it doesn't 
	have permissions. 
	http://msdn.microsoft.com/en-us/library/x8x7sakw(VS.80).aspx */

    char *parms_fn = _tempnam (0, "plastimatch_");
    FILE *fp = fopen (parms_fn, "wb+");
    printf ("parms_fn = %s\n", parms_fn);
# else
    FILE *fp = tmpfile ();
# endif
    return fp;
}

void
trim_trailing_slashes (char *pathname)
{
    char *p = pathname + strlen (pathname) - 1;
    while (p >= pathname && ISSLASH(*p)) {
	*p = 0;
    }
}

/* Caller must free memory */
char*
file_util_parent (const char *filename)
{
    char *tmp = 0;
    char *p = 0, *q = 0;

    if (!filename) return tmp;

    p = tmp = strdup (filename);
    trim_trailing_slashes (p);
    while (*p) {
	if (ISSLASH (*p)) {
	    q = p;
	}
	p ++;
    }
    if (q) {
	*q = 0;
	return tmp;
    } else {
	/* No directory separators -- return "." */
	free (tmp);
	return strdup (".");
    }
}

/* Caller must free memory */
char*
file_util_dirname (const char *filename)
{
    if (!filename) return 0;

    if (is_directory (filename)) {
	return strdup (filename);
    }

    return file_util_parent (filename);
}

void
strip_extension (char* filename)
{
    char *p;

    p = strrchr (filename, '.');
    if (p) {
	*p = 0;
    }
}

Pstring*
file_load (const char* filename)
{
    FILE *fp;
    Pstring *buf;

    fp = fopen (filename, "rb");
    if (!fp) {
	return 0;
    }

    /* Slurp the file into the buffer */
    buf = new Pstring ();
    buf->read ((bNread) fread, fp);
    fclose (fp);

    for (int i = 0; i < 20; i++) {
	printf ("%c", (char) (*buf)[i]);
    }
    printf ("\n");

    return buf;
}
