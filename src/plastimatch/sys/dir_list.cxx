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

Dir_list::Dir_list ()
{
    this->init ();
}

Dir_list::~Dir_list ()
{
    int i;
    if (this->entries) {
	for (i = 0; i < this->num_entries; i++) {
	    free (this->entries[i]);
	}
	free (this->entries);
    }
}

void
Dir_list::init ()
{
    this->num_entries = 0;
    this->entries = 0;
}

void 
Dir_list::load (const char* dir)
{
#if (_WIN32)
    intptr_t srch;
    struct _finddata_t d;
    char* buf;
    
    buf = (char*) malloc (strlen (dir) + 3);
    sprintf (buf, "%s/*", dir);

    srch = _findfirst (buf, &d);
    free (buf);

    if (srch == -1) return 0;

    do {
	this->num_entries ++;
	this->entries = (char**) realloc (
	    this->entries, 
	    this->num_entries * sizeof (char*));
	this->entries[this->num_entries-1] = strdup (d.name);
    } while (_findnext (srch, &d) != -1);
    _findclose (srch);

#else
    DIR* dp;
    struct dirent* d;

    dp = opendir (dir);
    if (!dp) return;

    for (d = readdir(dp); d; d = readdir(dp)) {
	this->num_entries ++;
	this->entries = (char**) realloc (
	    this->entries, 
	    this->num_entries * sizeof (char*));
	this->entries[this->num_entries-1] = strdup (d->d_name);
    }
    closedir (dp);
#endif
}
