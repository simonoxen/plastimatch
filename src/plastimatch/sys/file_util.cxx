/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"
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

#include "plmsys.h"

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

#if 0
void
make_directory_recursive (const Pstring& filename)
{
    make_directory_recursive (filename.c_str());
}
#endif

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

std::string
compose_filename (const char *a, const char *b)
{
    std::string output_fn;

    char *tmp = strdup (a);
    trim_trailing_slashes (tmp);
    output_fn = tmp;
    free (tmp);
    output_fn.append ("/");
    output_fn.append (b);
    return output_fn;
}

/* cross platform getcwd */
char*
plm_getcwd (char* s, int len)
{
#if (UNIX)
    return getcwd (s, len);
#elif (WIN32)
    /* Microsoft declares POSIX as "deprecated" */
    return _getcwd (s, len);
#endif
}

/* cross platform chdir */
int
plm_chdir (char* s)
{
#if (UNIX)
    return chdir (s);
#elif (WIN32)
    /* Microsoft declares POSIX as "deprecated" */
    return _chdir (s);
#endif
}

/* cross platform directory list */
int
plm_get_dir_list (const char*** f_list)
{
#if (WIN32)
    // Win32 Version goes here
    return -1;
#elif (UNIX)
    DIR *dp;
    struct dirent *ep;
    char b[NAME_MAX];
    int n;

    if (!plm_getcwd (b, NAME_MAX)) {
        return -1;
    }

    dp = opendir (b);
    if (dp != NULL) {
        n=0;
        while ((ep=readdir(dp))) { n++; }
        *f_list = (const char**)malloc (n*sizeof(char*));
        rewinddir (dp);
        n=0;
        while ((ep=readdir(dp))) {
            (*f_list)[n] = (char*)malloc (strlen(ep->d_name));
            strcpy ((char*)(*f_list)[n++], ep->d_name);
        }
        (void) closedir (dp);
        return n;
    } else {
        return -1;
    }
#endif
}
