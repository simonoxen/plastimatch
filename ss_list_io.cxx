/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bstring_util.h"
#include "file_util.h"
#include "plm_image_header.h"
#include "rtss.h"
#include "ss_list_io.h"

Rtss*
ss_list_load (Rtss* cxt, const char* ss_list_fn)
{
    FILE* fp;

    fp = fopen (ss_list_fn, "r");
    if (!fp) {
	print_and_exit (
	    "Could not open ss_list file for read: %s\n", ss_list_fn);
    }
    if (!cxt) {
	cxt = new Rtss;
    }

    /* Part 2: Structures info */
    while (1) {
        char color[CXT_BUFLEN];
        char name[CXT_BUFLEN];
        char buf[CXT_BUFLEN];
	int struct_id;
        char *p;
        int rc;

        p = fgets (buf, CXT_BUFLEN, fp);
        if (!p) {
	    break;
        }
        rc = sscanf (buf, "%d|%[^|]|%[^\r\n]", &struct_id, color, name);
        if (rc != 3) {
            fprintf (stderr, 
		"Error. ss_list file not formatted correctly: %s\n",
		ss_list_fn);
            exit (-1);
        }

	Cxt_structure *curr_structure = cxt->add_structure (
	    CBString (name), CBString (color), struct_id);
	curr_structure->bit = struct_id;
    }

    fclose (fp);
    return cxt;
}

void
ss_list_save (Rtss* cxt, const char* ss_list_fn)
{
    int i;
    FILE *fp;
	
    make_directory_recursive (ss_list_fn);
    fp = fopen (ss_list_fn, "w");
    if (!fp) {
	print_and_exit (
	    "Could not open ss_list file for write: %s\n", ss_list_fn);
    }

    for (i = 0; i < cxt->num_structures; i++) {
	Cxt_structure *curr_structure;
	curr_structure = &cxt->slist[i];
	fprintf (fp, "%d|%s|%s\n", 
	    curr_structure->bit, 
	    (bstring_empty (curr_structure->color) 
		? "255\\0\\0"
		: (const char*) curr_structure->color),
	    (const char*) curr_structure->name);
    }
    fclose (fp);
    printf ("Done.\n");
}

void
ss_list_save_colormap (Rtss* cxt, const char* colormap_fn)
{
    int i;
    FILE *fp;
	
    make_directory_recursive (colormap_fn);
    fp = fopen (colormap_fn, "w");
    if (!fp) {
	print_and_exit (
	    "Could not open colormap file for write: %s\n", colormap_fn);
    }

    fprintf (fp, 
	"# Color table file %s\n"
	"# %d values\n",
	colormap_fn, cxt->num_structures);

    for (i = 0; i < cxt->num_structures; i++) {
	int r, g, b;
	Cxt_structure *curr_structure;
	curr_structure = &cxt->slist[i];
	cxt_structure_rgb (curr_structure, &r, &g, &b);
	fprintf (fp, "%d %s %d %d %d 255\n", 
	    curr_structure->id, 
	    (const char*) curr_structure->name, 
	    r, g, b);
    }
    fclose (fp);
}
