/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include "cxt_io.h"
#include "cxt_to_mha.h"
#include "file_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_path.h"
#include "readmha.h"
#include "render_polyline.h"
#include "volume.h"

#if defined (commentout)
#if defined (WIN32)
#include <direct.h>
#define snprintf _snprintf
#define mkdir(a, b) _mkdir (a)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif
#endif


void
cxt_to_mha_init (
    Cxt_to_mha_state *ctm_state,
    Cxt_structure_list *structures,
    bool want_prefix_imgs,
    bool want_labelmap,
    bool want_ss_img
)
{
    int slice_voxels;

    slice_voxels = structures->dim[0] * structures->dim[1];

    ctm_state->want_prefix_imgs = want_prefix_imgs;
    ctm_state->want_labelmap = want_labelmap;
    ctm_state->want_ss_img = want_ss_img;

    ctm_state->acc_img = (unsigned char*) malloc (
	slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
       structure */
    ctm_state->uchar_vol 
	= volume_create (structures->dim, structures->offset, 
	    structures->spacing, PT_UCHAR, 0, 0);
    if (ctm_state->uchar_vol == 0) {
	print_and_exit ("ERROR: failed in allocating the volume");
    }

    /* Create output volume for labelmap */
    ctm_state->labelmap_vol = 0;
    if (want_labelmap) {
	ctm_state->labelmap_vol = volume_create (structures->dim, 
	    structures->offset, structures->spacing, PT_UINT32, 0, 0);
	if (ctm_state->labelmap_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Create output volume for ss_img */
    ctm_state->ss_img_vol = 0;
    if (want_ss_img) {
	ctm_state->ss_img_vol = volume_create (structures->dim, 
	    structures->offset, structures->spacing, PT_UINT32, 0, 0);
	if (ctm_state->ss_img_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Initialize to start with first structure */
    ctm_state->curr_struct_no = 0;
}

/* Return true if an image was processed */
bool
cxt_to_mha_process_next (
    Cxt_to_mha_state *ctm_state,
    Cxt_structure_list *structures
)
{
    Cxt_structure* curr_structure;
    Cxt_polyline* curr_contour;
    unsigned char* uchar_img = (unsigned char*) ctm_state->uchar_vol->img;
    int slice_voxels;

    /* If done, return false */
    if (ctm_state->curr_struct_no >= structures->num_structures) {
	ctm_state->curr_struct_no = structures->num_structures + 1;
	return false;
    }
    
    curr_structure = &structures->slist[ctm_state->curr_struct_no];
    slice_voxels = structures->dim[0] * structures->dim[1];

    memset (uchar_img, 0, structures->dim[0] * structures->dim[1] 
		* structures->dim[2] * sizeof(unsigned char));

    /* Loop through polylines in this structure */
    for (int i = 0; i < curr_structure->num_contours; i++) {
	unsigned char* uchar_slice;

	curr_contour = &curr_structure->pslist[i];
	if (curr_contour->slice_no == -1) {
	    continue;
	}

	/* Render contour to binary */
	memset (ctm_state->acc_img, 0, slice_voxels * sizeof(unsigned char));
	render_slice_polyline (ctm_state->acc_img, 
	    structures->dim, 
	    structures->spacing, 
	    structures->offset,
	    curr_contour->num_vertices, 
	    curr_contour->x, curr_contour->y);

	/* Copy from acc_img into mask image */
	if (ctm_state->want_prefix_imgs) {
	    uchar_slice = &uchar_img[curr_contour->slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		uchar_slice[k] ^= ctm_state->acc_img[k];
	    }
	}

	/* Copy from acc_img into labelmask and xormap images */
	if (ctm_state->want_labelmap) {
	    uint32_t* labelmap_img;
	    uint32_t* uint32_slice;
	    labelmap_img = (uint32_t*) ctm_state->labelmap_vol->img;
	    uint32_slice = &labelmap_img[curr_contour->slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		if (ctm_state->acc_img[k]) {
		    uint32_slice[k] = ctm_state->curr_struct_no + 1;
		}
	    }
	}
	if (ctm_state->want_ss_img) {
	    uint32_t* ss_img_img = 0;
	    uint32_t* uint32_slice;
	    ss_img_img = (uint32_t*) ctm_state->ss_img_vol->img;
	    uint32_slice = &ss_img_img[curr_contour->slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		if (ctm_state->acc_img[k]) {
		    uint32_slice[k] |= (1 << ctm_state->curr_struct_no);
		}
	    }
	}
    }

    ctm_state->curr_struct_no ++;
    return true;
}

const char*
cxt_to_mha_current_name (
    Cxt_to_mha_state *ctm_state,
    Cxt_structure_list *structures
)
{
    if (ctm_state->curr_struct_no == 0 
	|| ctm_state->curr_struct_no == structures->num_structures + 1)
    {
	Cxt_structure *curr_structure;
	curr_structure = &structures->slist[ctm_state->curr_struct_no-1];
	return curr_structure->name;
    } else {
	return "";
    }
}

void
cxt_to_mha_free (Cxt_to_mha_state *ctm_state)
{
    if (ctm_state->uchar_vol) {
	volume_destroy (ctm_state->uchar_vol);
    }
    if (ctm_state->labelmap_vol) {
	volume_destroy (ctm_state->labelmap_vol);
    }
    if (ctm_state->ss_img_vol) {
	volume_destroy (ctm_state->ss_img_vol);
    }
    free (ctm_state->acc_img);
}


#if defined (commentout)
int
cxt_to_mha_
    fixed,
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

    structures = (Cxt_structure_list*) malloc (sizeof(Cxt_structure_list));
    curr_structure = (Cxt_structure*) malloc (sizeof(Cxt_structure));
    memset (structures, 0, sizeof(Cxt_structure_list));
    structures->num_structures = 0;
    memset (curr_structure, 0, sizeof(Cxt_structure));
    curr_structure->num_contours = 0;

    cxt_read (structures, parms->cxt_fn);

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
#endif
