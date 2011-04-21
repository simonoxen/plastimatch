/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dir_list.h"
#include "file_util.h"
#include "plm_path.h"
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
    Proj_image_dir *pid
)
{
    int i;

    /* Search for appropriate entry */
    for (i = 0; i < pid->num_proj_images; i++) {
	char *entry = pid->proj_image_list[i];
	int rc;
	char prefix[2048], num[2048];

	/* Look for prefix + number at beginning of filename */
	rc = sscanf (entry, "%2047[^0-9]%2047[0-9]", prefix, num);
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
	    pid->img_pat = (char*) malloc (
		strlen (pid->dir) + 1 + strlen (prefix) 
		+ strlen (num_pat) + strlen (suffix) + 1);
	    sprintf (pid->img_pat, "%s/%s%s%s", pid->dir, 
		prefix, num_pat, suffix);

	    /* Done! */
	    break;
	}
    }
    return;
}

static void
proj_image_dir_load_filenames (
    Proj_image_dir *pid,
    char *dir
)
{
    int i;
    Dir_list *dir_list;

    if (pid->dir) {
	free (pid->dir);
	pid->dir = 0;
    }

    dir_list = dir_list_load (0, dir);
    if (!dir_list) {
	return;
    }

    pid->dir = strdup (dir);
    pid->num_proj_images = 0;
    pid->proj_image_list = 0;

    for (i = 0; i < dir_list->num_entries; i++) {
	char *entry = dir_list->entries[i];
	if (extension_is (entry, ".hnd") || extension_is (entry, ".pfm") 
	    || extension_is (entry, ".raw"))
	{
	    pid->num_proj_images ++;
	    pid->proj_image_list = (char**) realloc (
		pid->proj_image_list,
		pid->num_proj_images * sizeof (char*));
	    pid->proj_image_list[pid->num_proj_images-1] = strdup (entry);
	}
    }

    dir_list_destroy (dir_list);
}

static void
proj_image_dir_harden_filenames (
    Proj_image_dir *pid
)
{
    int i;

    for (i = 0; i < pid->num_proj_images; i++) {
	char img_file[_MAX_PATH];
	char *entry = pid->proj_image_list[i];
	snprintf (img_file, _MAX_PATH, "%s/%s", pid->dir, entry);
	pid->proj_image_list[i] = strdup (img_file);
	free (entry);
    }
}

static void
proj_image_dir_clear_filenames (
    Proj_image_dir *pid
)
{
    int i;
    for (i = 0; i < pid->num_proj_images; i++) {
	char *entry = pid->proj_image_list[i];
	free (entry);
    }
    if (pid->proj_image_list) free (pid->proj_image_list);
    pid->num_proj_images = 0;
    pid->proj_image_list = 0;
}

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
Proj_image_dir*
proj_image_dir_create (char *dir)
{
    Proj_image_dir *pid;
    char xml_file[_MAX_PATH];

    pid = (Proj_image_dir*) malloc (sizeof (Proj_image_dir));
    memset (pid, 0, sizeof (Proj_image_dir));

    /* Look for ProjectionInfo.xml */
    snprintf (xml_file, _MAX_PATH, "%s/%s", dir, "ProjectionInfo.xml");
    if (file_exists (xml_file)) {
	pid->xml_file = strdup (xml_file);
    }

    /* Load list of file names */
    proj_image_dir_load_filenames (pid, dir);

    /* If base directory doesn't contain images, look in Scan0 directory */
    if (pid->num_proj_images == 0) {
	char scan0_dir[_MAX_PATH];
	snprintf (scan0_dir, _MAX_PATH, "%s/%s", dir, "Scan0");

	/* Load list of file names */
	proj_image_dir_load_filenames (pid, scan0_dir);
    }

    /* No images in either base directory or Scan 0, so give up. */
    if (pid->num_proj_images == 0) {
	proj_image_dir_destroy (pid);
	return 0;
    }

    /* Found images, try to find pattern */
    proj_image_dir_find_pattern (pid);

    /* Convert relative paths to absolute paths */
    proj_image_dir_harden_filenames (pid);

    return pid;
}

void
proj_image_dir_select (Proj_image_dir *pid, int first, int skip, int last)
{
    int i;

    if (!pid || pid->num_proj_images == 0 || !pid->img_pat) {
	return;
    }

    proj_image_dir_clear_filenames (pid);
    for (i = first; i <= last; i += skip) {
	char img_file[_MAX_PATH];
	snprintf (img_file, _MAX_PATH, pid->img_pat, i);
	if (file_exists (img_file)) {
	    pid->num_proj_images ++;
	    pid->proj_image_list = (char**) realloc (
		pid->proj_image_list, 
		pid->num_proj_images * sizeof (char*));
	    pid->proj_image_list[pid->num_proj_images-1] = strdup (img_file);
	}
    }
}


void
proj_image_dir_destroy (Proj_image_dir *pid)
{
    if (pid->img_pat) free (pid->img_pat);
    if (pid->xml_file) free (pid->xml_file);
    proj_image_dir_clear_filenames (pid);

    free (pid);
}

Proj_image* 
proj_image_dir_load_image (Proj_image_dir* pid, int index)
{
    if (index < 0 || index >= pid->num_proj_images) {
	return 0;
    }

    /* mat file load not yet implemented -- only works for hnd files */
    return proj_image_load (pid->proj_image_list[index], 0);
}
