/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "plm_config.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_path.h"
#include "volume.h"
#include "cxt_io.h"
#include "file_util.h"
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
    char xormap_fn[_MAX_PATH];
    char xorlist_fn[_MAX_PATH];
    char fixed_fn[_MAX_PATH];
    char* cxt_fn;
    char prefix[_MAX_PATH];
};

void
print_usage (void)
{
    printf ("Usage: cxt_to_mha [options] cxt_file\n");
    printf ("  The cxt_file is an ASCII file with the contours\n");
    printf ("  The prefix is (e.g.) a 4 digit patient number.\n");
    printf ("Options:\n");
    printf ("  --prefix   string       Generate one file per structure with prefix\n");
    printf ("  --xormap   filename     Generate multi-structure map\n");
    printf ("  --xorlist  filename     File with xormap structure names\n");
    printf ("  --labelmap filename     Generate Slicer3 labelmap\n");
    printf ("  --fixed filename        Render mha with fixed resolution and geometry\n");
    exit (-1);
}

void
parse_args (Program_Parms* parms, int argc, char* argv[])
{
    int ch;
    static struct option longopts[] = {
	{ "labelmap",       required_argument,      NULL,           1 },
	{ "xormap",         required_argument,      NULL,           2 },
	{ "xorlist",        required_argument,      NULL,           3 },
	{ "prefix",         required_argument,      NULL,           4 },
	{ "fixed",          required_argument,      NULL,           5 },
	{ NULL,             0,                      NULL,           0 }
    };

    parms->labelmap_fn[0] = 0;
    parms->xormap_fn[0] = 0;
    parms->xorlist_fn[0] = 0;
    parms->fixed_fn[0] = 0;
    parms->prefix[0] = 0;

    while ((ch = getopt_long (argc, argv, "", longopts, NULL)) != -1) {
	switch (ch) {
	case 1:
	    strncpy (parms->labelmap_fn, optarg, _MAX_PATH);
	    break;
	case 2:
	    strncpy (parms->xormap_fn, optarg, _MAX_PATH);
	    break;
	case 3:
	    strncpy (parms->xorlist_fn, optarg, _MAX_PATH);
	    break;
	case 4:
	    strncpy (parms->prefix, optarg, _MAX_PATH);
	    break;
	case 5:
	    strncpy (parms->fixed_fn, optarg, _MAX_PATH);
	    break;
	default:
	    break;
	}
    }

    argc -= optind;
    argv += optind;

    if (argc != 1) {
	print_usage ();
    }

    parms->cxt_fn = argv[0];
}

int
main (int argc, char* argv[])
{
    Program_Parms* parms = (Program_Parms*) malloc (sizeof(Program_Parms));
    Cxt_structure_list* structures;
    Cxt_structure* curr_structure;
    Cxt_polyline* curr_contour;
    Volume* uchar_vol;
    Volume* labelmap_vol = 0;
    Volume* xormap_vol = 0;

    unsigned char* uchar_img;
    uint32_t* labelmap_img = 0;
    uint32_t* xormap_img = 0;
    unsigned char* acc_img;
    int dim[2];
    float offset[2];
    float spacing[2];
    int slice_voxels = 0;
    int sno = 0;		/* Structure number */

    FILE *xorlist_fp = 0;

    parse_args (parms, argc, argv);

    structures = (Cxt_structure_list*) malloc (sizeof(Cxt_structure_list));
    curr_structure = (Cxt_structure*) malloc (sizeof(Cxt_structure));
    memset (structures, 0, sizeof(Cxt_structure_list));
    structures->num_structures = 0;
    memset (curr_structure, 0, sizeof(Cxt_structure));
    curr_structure->num_contours = 0;

    cxt_load (structures, parms->cxt_fn);

    /* Override cxt geometry if user specified --fixed */
    if (parms->fixed_fn[0]) {
	FloatImageType::Pointer fixed = load_float (parms->fixed_fn, 0);
	PlmImageHeader pih;
	
	pih.set_from_itk_image (fixed);
	pih.get_gpuit_origin (structures->offset);
	pih.get_gpuit_spacing (structures->spacing);
	pih.get_gpuit_dim (structures->dim);
	
    }

    dim[0] = structures->dim[0];
    dim[1] = structures->dim[1];
    offset[0] = structures->offset[0];
    offset[1] = structures->offset[1];
    spacing[0] = structures->spacing[0];
    spacing[1] = structures->spacing[1];
    slice_voxels = dim[0] * dim[1];

    if (parms->xorlist_fn[0]) {
	make_directory_recursive (parms->xorlist_fn);
	xorlist_fp = fopen (parms->xorlist_fn, "w");
	if (!xorlist_fp) {
	    fprintf (stderr, "Error opening file for write: %s\n",
		     parms->xorlist_fn);
	    exit (-1);
	}
    }

    acc_img = (unsigned char*) malloc (slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
       structure */
    uchar_vol = volume_create (structures->dim, structures->offset, 
			       structures->spacing, PT_UCHAR, 0, 0);
    if (uchar_vol == 0) {
	fprintf (stderr, "ERROR: failed in allocating the volume");
    }
    uchar_img = (unsigned char*) uchar_vol->img;

    /* Create output volume for labelmask and xormap image. */
    if (parms->labelmap_fn[0]) {
	labelmap_vol = volume_create (structures->dim, structures->offset, 
				      structures->spacing, PT_UINT32, 0, 0);
	if (labelmap_vol == 0) {
	    fprintf (stderr, "ERROR: failed in allocating the volume");
	}
	labelmap_img = (uint32_t*) labelmap_vol->img;
    }
    if (parms->xormap_fn[0]) {
	xormap_vol = volume_create (structures->dim, structures->offset, 
				    structures->spacing, PT_UINT32, 0, 0);
	if (xormap_vol == 0) {
	    fprintf (stderr, "ERROR: failed in allocating the volume");
	}
	xormap_img = (uint32_t*) xormap_vol->img;
    }

    for (int j = 0; j < structures->num_structures; j++) {
	curr_structure = &structures->slist[j];
	char fn[BUFLEN] = "";

	if (parms->prefix[0]) {
	    strcat (fn, parms->prefix);
	    strcat (fn, "_");
	    strcat (fn, curr_structure->name);
	    strcat (fn, ".mha");
	}

	memset (uchar_img, 0, structures->dim[0] * structures->dim[1] 
		* structures->dim[2] * sizeof(unsigned char));

	if (curr_structure->num_contours == 0) {
	    continue;
	}

	for (int i = 0; i < curr_structure->num_contours; i++) {
	    unsigned char* uchar_slice;

	    curr_contour = &curr_structure->pslist[i];

	    if (curr_contour->slice_no == -1) {
		continue;
	    }

	    /* Render contour to binary */
	    memset (acc_img, 0, dim[0] * dim[1] * sizeof(unsigned char));
	    render_slice_polyline (acc_img, dim, spacing, offset,
				   curr_contour->num_vertices, 
				   curr_contour->x, curr_contour->y);

	    /* Copy from acc_img into mask image */
	    if (parms->prefix[0]) {
		uchar_slice = &uchar_img[curr_contour->slice_no * dim[0] * dim[1]];
		for (int k = 0; k < slice_voxels; k++) {
		    uchar_slice[k] ^= acc_img[k];
		}
	    }

	    /* Copy from acc_img into labelmask and xormap images */
	    if (parms->labelmap_fn[0]) {
		uint32_t* uint32_slice;
		uint32_slice = &labelmap_img[curr_contour->slice_no * dim[0] * dim[1]];
		for (int k = 0; k < slice_voxels; k++) {
		    if (acc_img[k]) {
			uint32_slice[k] = sno + 1;
		    }
		}
	    }
	    if (parms->xormap_fn[0]) {
		uint32_t* uint32_slice;
		uint32_slice = &xormap_img[curr_contour->slice_no * dim[0] * dim[1]];
		for (int k = 0; k < slice_voxels; k++) {
		    if (acc_img[k]) {
			uint32_slice[k] |= (1 << sno);
		    }
		}
	    }
	}

	if (parms->xorlist_fn[0]) {
	    fprintf (xorlist_fp, "%d|%s|%s\n",
		     sno, 
		     (curr_structure->color 
		      ? (const char*) curr_structure->color->data 
		      : "\255\\0\\0"),
		     curr_structure->name);
	}

	if (parms->prefix[0]) {
	    printf ("writing file: %s\n", fn);
	    write_mha (fn, uchar_vol);
	}
	sno ++;
    }
    if (parms->labelmap_fn[0]) {
	printf ("writing file: %s\n", parms->labelmap_fn);
	write_mha (parms->labelmap_fn, labelmap_vol);
	volume_destroy (labelmap_vol);
    }
    if (parms->xormap_fn[0]) {
	printf ("writing file: %s\n", parms->xormap_fn);
	write_mha (parms->xormap_fn, xormap_vol);
	volume_destroy (xormap_vol);
    }
    if (parms->xorlist_fn[0]) {
	printf ("writing file: %s\n", parms->xorlist_fn);
	fclose (xorlist_fp);
    }
    if (parms->prefix[0]) {
	volume_destroy (uchar_vol);
    }
    free (parms);
}
