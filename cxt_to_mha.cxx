/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "plm_path.h"
#include "volume.h"
#include "readcxt.h"
#include "readmha.h"

#if defined (WIN32)
#include <direct.h>
#define snprintf _snprintf
#define mkdir(a, b) _mkdir (a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include "render_polyline.h"
#include "getopt.h"

#define BUFLEN 2048
//#define BUF (128 * 1024)

typedef struct program_parms Program_Parms;
struct program_parms {
    char labelmap_fn[_MAX_PATH];
    char* cxt_fn;
    char* prefix;
};

void
print_usage (void)
{
    printf ("Usage: cxt_to_mha [options] cxt_file prefix\n");
    printf ("  The cxt_file is an ASCII file with the contours\n");
    printf ("  The prefix is (e.g.) a 4 digit patient number.\n");
    printf ("Options:\n");
    printf ("  --labelmap filename     Generate Slicer3 labelmap\n");
    exit (-1);
}

void
parse_args (Program_Parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "labelmap",       required_argument,      NULL,           1 },
	{ NULL,             0,                      NULL,           0 }
    };

    parms->labelmap_fn[0] = 0;

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->labelmap_fn, optarg, _MAX_PATH);
	    break;
	default:
	    break;
	}
    }

    argc -= optind;
    argv += optind;

    if (argc < 1 || argc > 2) {
	print_usage ();
    }

    parms->cxt_fn = argv[0];
    if (argc == 2) {
	parms->prefix = argv[1];
    } else {
	parms->prefix = 0;
    }
}

int
main (int argc, char* argv[])
{
    Program_Parms* parms = (Program_Parms*) malloc (sizeof(Program_Parms));
    STRUCTURE_List* structures;
    STRUCTURE* curr_structure;
    POLYLINE* curr_contour;
    Volume* uchar_vol;
    Volume* ulong_vol;

    unsigned char* uchar_img;
    unsigned long* ulong_img;
    unsigned char* acc_img;
    int dim[2];
    float offset[2];
    float spacing[2];
    int slice_voxels = 0;

    parse_args (parms, argc, argv);

    structures = (STRUCTURE_List*) malloc (sizeof(STRUCTURE_List));
    curr_structure = (STRUCTURE*) malloc (sizeof(STRUCTURE));
    memset (structures, 0, sizeof(STRUCTURE_List));
    structures->num_structures = 0;
    memset (curr_structure, 0, sizeof(STRUCTURE));
    curr_structure->num_contours = 0;

    cxt_read (structures, parms->cxt_fn);

    dim[0] = structures->dim[0];
    dim[1] = structures->dim[1];
    offset[0] = structures->offset[0];
    offset[1] = structures->offset[1];
    spacing[0] = structures->spacing[0];
    spacing[1] = structures->spacing[1];
    slice_voxels = dim[0] * dim[1];

    acc_img = (unsigned char*) malloc (slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
	structure */
    uchar_vol = volume_create (structures->dim, structures->offset, 
			 structures->spacing, PT_UCHAR, 0, 0);
    if (uchar_vol == 0) {
        fprintf (stderr, "ERROR: failed in allocating the volume");
    }
    uchar_img = (unsigned char*) uchar_vol->img;

    /* Create output volume for labelmask image. */
    if (parms->labelmap_fn[0]) {
	ulong_vol = volume_create (structures->dim, structures->offset, 
			     structures->spacing, PT_ULONG, 0, 0);
	if (ulong_vol == 0) {
	    fprintf (stderr, "ERROR: failed in allocating the volume");
	}
	ulong_img = (unsigned long*) ulong_vol->img;
    }

    for (int j = 0; j < structures->num_structures; j++) {
        curr_structure = &structures->slist[j];
        char fn[BUFLEN] = "";

	if (parms->prefix) {
	    strcat (fn, parms->prefix);
	    strcat (fn, "_");
	    strcat (fn, curr_structure->name);
	    strcat (fn, ".mha");
	}

	memset (uchar_img, 0, structures->dim[0] * structures->dim[1] 
		* structures->dim[2] * sizeof(unsigned char));

	for (int i = 0; i < curr_structure->num_contours; i++) {
            unsigned char* uchar_slice;
            unsigned long* ulong_slice;

	    curr_contour = &curr_structure->pslist[i];
            printf ("Slice# %3d\n", curr_contour->slice_no);

            memset (acc_img, 0, dim[0] * dim[1] * sizeof(unsigned char));
            render_slice_polyline (acc_img, dim, spacing, offset,
                                   curr_contour->num_vertices, 
				   curr_contour->x, curr_contour->y);

	    /* Copy from acc_img into mask image */
	    if (parms->prefix) {
		uchar_slice = &uchar_img[curr_contour->slice_no * dim[0] * dim[1]];
		for (int k = 0; k < slice_voxels; k++) {
		    uchar_slice[k] ^= acc_img[k];
		}
	    }

	    /* Copy from acc_img into labelmask image */
	    if (parms->labelmap_fn[0]) {
		ulong_slice = &ulong_img[curr_contour->slice_no * dim[0] * dim[1]];
		for (int k = 0; k < slice_voxels; k++) {
		    if (acc_img[k]) {
			ulong_slice[k] = j + 1;
		    }
		}
	    }
        }
	if (parms->prefix) {
	    printf ("writing file: %s\n", fn);
	    write_mha (fn, uchar_vol);
	}
    }
    if (parms->labelmap_fn[0]) {
	printf ("writing file: %s\n", parms->labelmap_fn);
	write_mha (parms->labelmap_fn, ulong_vol);
	volume_free (ulong_vol);
    }
    if (parms->prefix) {
	volume_free (uchar_vol);
    }
    free (parms);
}
