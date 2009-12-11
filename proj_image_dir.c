/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dir_list.h"
#include "file_util.h"
#include "proj_image_dir.h"

/* -----------------------------------------------------------------------
   Private functions
   ----------------------------------------------------------------------- */

/* This utility function tries to guess the pattern of filenames 
   within a directory.  The patterns are of the form: 
   
     "XXXXYYYY.ZZZ"
     
   where XXXX is a prefix, YYYY is a number, and .ZZZ is the extension 
   of a known type (either .hnd, .pfm, or .raw).   Returns a pattern
   which can be used with sprintf, such as:

     "XXXX%04d.ZZZ"

   Caller must free patterns.
*/
static void
proj_image_dir_find_pattern (
    char **img_pat, 
    char **mat_pat, 
    char *dir)
{
    int i;
    Dir_list *dir_list;

    *mat_pat = 0;
    *img_pat = 0;

    /* Load directory */
    dir_list = dir_list_load (0, dir);
    if (!dir_list) return;

    /* Search for appropriate entry */
    for (i = 0; i < dir_list->num_entries; i++) {
	char *entry = dir_list->entries[i];
	if (extension_is (entry, ".hnd") || extension_is (entry, ".pfm") 
	    || extension_is (entry, ".raw"))
	{
	    int rc;
	    char prefix[2048], num[2048];
	    rc = sscanf (entry, "%2047[^0-9]%2047[0-9]", prefix, num);

	    /* Found entry */
	    if (rc == 2) {
		char num_pat[5];
		char *suffix;

		/* Two cases: if num has a leading 0, we format such as 
		   %05d.  If, we format as %d. */
		if (num[0] == '0') {
		    strcpy (num_pat, "%0_d");
		    num_pat[2] = '0' + strlen (num);
		} else {
		    strcpy (num_pat, "%d");
		}

		/* Find suffix */
		suffix = &entry[strlen(prefix) + strlen(num)];
		
		/* Create pattern */
		*img_pat = (char*) malloc (
		    strlen (dir) + 1 + strlen (prefix) 
		    + strlen (num_pat) + strlen (suffix) + 1);
		*mat_pat = (char*) malloc (
		    strlen (dir) + 1 + strlen (prefix) 
		    + strlen (num_pat) + 4 + 1);
		sprintf (*img_pat, "%s/%s%s%s", dir, prefix, num_pat, suffix);
		sprintf (*mat_pat, "%s/%s%s%s", dir, prefix, num_pat, ".txt");

		/* Done! */
		break;
	    }
	}
    }
    dir_list_destroy (dir_list);
    return;
}

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
Proj_image_dir*
proj_image_dir_create (char *dir)
{
    Proj_image_dir *pid;
    pid = (Proj_image_dir*) malloc (sizeof (Proj_image_dir));
    
    proj_image_dir_find_pattern (&pid->img_pat, &pid->mat_pat, dir);
    if (!pid->img_pat) {
	proj_image_dir_destroy (pid);
	return 0;
    }
    return pid;
}

void
proj_image_dir_destroy (Proj_image_dir *pid)
{
    if (pid->img_pat) free (pid->img_pat);
    if (pid->mat_pat) free (pid->mat_pat);
    free (pid);
}

Proj_image* 
proj_image_dir_load_image (Proj_image_dir* pid, int index)
{
    char img_file[1024], mat_file[1024];

    snprintf (img_file, 1024, pid->img_pat, index);
    snprintf (mat_file, 1024, pid->mat_pat, index);
    return proj_image_load (img_file, mat_file);
}
