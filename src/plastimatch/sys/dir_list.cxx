/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmsys_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if (defined(_WIN32) || defined(WIN32))
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#endif

#include "plmsys.h"

#include "dir_list.h"

Dir_list*
dir_list_create (void)
{
    Dir_list *dl;
    dl = (Dir_list*) malloc (sizeof (Dir_list));
    dir_list_init (dl);
    return dl;
}

void
dir_list_init (Dir_list* dl)
{
    dl->num_entries = 0;
    dl->entries = 0;
}

/* Caller must free output.  Returns 0 if dir is not a directory. */
Dir_list *
dir_list_load (Dir_list *dir_list, const char* dir)
{
    Dir_list *dl;

#if (_WIN32)
    intptr_t srch;
    struct _finddata_t d;
    char* buf;
    
    buf = (char*) malloc (strlen (dir) + 3);
    sprintf (buf, "%s/*", dir);

    srch = _findfirst (buf, &d);
    free (buf);

    if (srch == -1) return 0;

    if (dir_list) {
	dl = dir_list;
    } else {
	dl = dir_list_create ();
    }

    do {
	dl->num_entries ++;
	dl->entries = (char**) realloc (
	    dl->entries, 
	    dl->num_entries * sizeof (char*));
	dl->entries[dl->num_entries-1] = strdup (d.name);
    } while (_findnext (srch, &d) != -1);
    _findclose (srch);

#else
    DIR* dp;
    struct dirent* d;

    dp = opendir (dir);
    if (!dp) return 0;

    if (dir_list) {
	dl = dir_list;
    } else {
	dl = dir_list_create ();
    }
    for (d = readdir(dp); d; d = readdir(dp)) {
	dl->num_entries ++;
	dl->entries = (char**) realloc (
	    dl->entries, 
	    dl->num_entries * sizeof (char*));
	dl->entries[dl->num_entries-1] = strdup (d->d_name);
    }
    closedir (dp);
#endif
    return dl;
}

void
dir_list_destroy (Dir_list *dir_list)
{
    int i;
    if (!dir_list) return;

    if (dir_list->entries) {
	for (i = 0; i < dir_list->num_entries; i++) {
	    free (dir_list->entries[i]);
	}
	free (dir_list->entries);
    }
    free (dir_list);
}
