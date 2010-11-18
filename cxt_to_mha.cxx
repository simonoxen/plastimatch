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
#include "mha_io.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_path.h"
#include "render_polyline.h"
#include "volume.h"

void
cxt_to_mha_init (
    Cxt_to_mha_state *ctm_state,       /* Output */
    Rtss_polyline_set *cxt,                         /* Input */
    bool want_prefix_imgs,             /* Input */
    bool want_labelmap,                /* Input */
    bool want_ss_img                   /* Input */
)
{
    int slice_voxels;

    slice_voxels = cxt->dim[0] * cxt->dim[1];

    ctm_state->want_prefix_imgs = want_prefix_imgs;
    ctm_state->want_labelmap = want_labelmap;
    ctm_state->want_ss_img = want_ss_img;

    ctm_state->acc_img = (unsigned char*) malloc (
	slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
       structure */
    ctm_state->uchar_vol 
	= volume_create (cxt->dim, cxt->offset, 
	    cxt->spacing, PT_UCHAR, 0, 0);
    if (ctm_state->uchar_vol == 0) {
	print_and_exit ("ERROR: failed in allocating the volume");
    }

    /* Create output volume for labelmap */
    ctm_state->labelmap_vol = 0;
    if (want_labelmap) {
	ctm_state->labelmap_vol = volume_create (cxt->dim, 
	    cxt->offset, cxt->spacing, PT_UINT32, 0, 0);
	if (ctm_state->labelmap_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Create output volume for ss_img */
    ctm_state->ss_img_vol = 0;
    if (want_ss_img) {
	ctm_state->ss_img_vol = volume_create (cxt->dim, 
	    cxt->offset, cxt->spacing, PT_UINT32, 0, 0);
	if (ctm_state->ss_img_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Initialize to start with first structure */
    ctm_state->curr_struct_no = 0;
    ctm_state->curr_bit = 0;
}

/* Return true if an image was processed */
bool
cxt_to_mha_process_next (
    Cxt_to_mha_state *ctm_state,       /* In/out */
    Rtss_polyline_set *cxt                          /* In/out */
)
{
    Rtss_structure* curr_structure;
    unsigned char* uchar_img = (unsigned char*) ctm_state->uchar_vol->img;
    int slice_voxels;

    /* If done, return false */
    if (ctm_state->curr_struct_no >= cxt->num_structures) {
	ctm_state->curr_struct_no = cxt->num_structures + 1;
	printf ("   --> (returning false)\n");
	return false;
    }
    
    curr_structure = cxt->slist[ctm_state->curr_struct_no];
    slice_voxels = cxt->dim[0] * cxt->dim[1];

    memset (uchar_img, 0, cxt->dim[0] * cxt->dim[1] 
		* cxt->dim[2] * sizeof(unsigned char));

    /* Loop through polylines in this structure */
    for (int i = 0; i < curr_structure->num_contours; i++) {
	Rtss_polyline* curr_contour;
	unsigned char* uchar_slice;

	curr_contour = curr_structure->pslist[i];
	if (curr_contour->slice_no == -1) {
	    continue;
	}

	/* Render contour to binary */
	memset (ctm_state->acc_img, 0, slice_voxels * sizeof(unsigned char));
	render_slice_polyline (
	    ctm_state->acc_img, 
	    cxt->dim, 
	    cxt->spacing, 
	    cxt->offset,
	    curr_contour->num_vertices, 
	    curr_contour->x, 
	    curr_contour->y);

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
		    uint32_slice[k] = ctm_state->curr_bit + 1;
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
		    uint32_slice[k] |= (1 << ctm_state->curr_bit);
		}
	    }
	}
    }

    ctm_state->curr_struct_no ++;
    if (curr_structure->num_contours > 0) {
	curr_structure->bit = ctm_state->curr_bit;
	ctm_state->curr_bit ++;
    }

    return true;
}

const char*
cxt_to_mha_current_name (
    Cxt_to_mha_state *ctm_state,
    Rtss_polyline_set *cxt
)
{
    if (ctm_state->curr_struct_no < cxt->num_structures + 1) {
	Rtss_structure *curr_structure;
	curr_structure = cxt->slist[ctm_state->curr_struct_no-1];
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

Cxt_to_mha_state*
cxt_to_mha_create (
    Rtss_polyline_set *cxt
)
{
    Cxt_to_mha_state *ctm_state = new Cxt_to_mha_state;
    cxt_to_mha_init (ctm_state, cxt, false, true, true);
    while (cxt_to_mha_process_next (ctm_state, cxt)) {}

    return ctm_state;
}

void
cxt_to_mha_destroy (Cxt_to_mha_state *ctm_state)
{
    cxt_to_mha_free (ctm_state);
    delete ctm_state;
}
