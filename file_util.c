/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#if HAVE_SYS_STAT
#include <sys/stat.h>
#endif
#if _WIN32
#include <direct.h>
#endif

int
file_exists (char *filename)
{
    FILE *f = fopen (filename, "r");
    if (f) {
	fclose (f);
	return 1;
    }
    return 0;
}

void
make_directory (char *dirname)
{
#if (_WIN32)
    mkdir (dirname);
#else
    mkdir (dirname, 0777);
#endif
}
