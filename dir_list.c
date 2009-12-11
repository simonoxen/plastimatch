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
#include "dir_list.h"
#include "print_and_exit.h"

Dir_list*
dir_list_create (void)
{
    Dir_list *dl;
    dl = (Dir_list*) malloc (sizeof (Dir_list));
    dl->num_entries = 0;
    dl->entries = 0;
    return dl;
}

/* Caller must free output.  Returns 0 if dir is not a directory. */
Dir_list *
dir_list_load (Dir_list *dir_list, const char* dir)
{
    Dir_list *dl;

#if (_WIN32)
    print_and_exit ("dir_list_load is unimplemented on windows\n");
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
