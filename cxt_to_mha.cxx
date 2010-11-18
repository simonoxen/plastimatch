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
#include "math_util.h"
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
    Rtss_polyline_set *cxt,            /* Input */
    bool want_prefix_imgs,             /* Input */
    bool want_labelmap,                /* Input */
    bool want_ss_img                   /* Input */
)
{
    int slice_voxels;

    /* GCS FIX: The rast_xxx are correct geometry for rasterization 
       in some, but not all circumstances. 

       We should use rast_xxx in the following cases:
       (a) Warping
       (b) No known output geometry
       (c) Output geometry doesn't match slice locations

       However, in the other cases we can directly convert to the 
       output geometry without using rast_xxx, then resampling
       to the output geometry.  These cases are common, such as 
       a simple conversion of CT+RTSS to MHA.
    */

    /* This code forces the second case.  We should delete this 
       code when the calling routine is smart enough to choose the 
       rasterization geometry. */
    for (int d = 0; d < 3; d++) {
	cxt->rast_offset[d] = cxt->offset[d];
	cxt->rast_spacing[d] = cxt->spacing[d];
	cxt->rast_dim[d] = cxt->dim[d];
    }

    slice_voxels = cxt->rast_dim[0] * cxt->rast_dim[1];

    ctm_state->want_prefix_imgs = want_prefix_imgs;
    ctm_state->want_labelmap = want_labelmap;
    ctm_state->want_ss_img = want_ss_img;

    ctm_state->acc_img = (unsigned char*) malloc (
	slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
       structure */
    ctm_state->uchar_vol 
	= volume_create (cxt->rast_dim, cxt->rast_offset, 
	    cxt->rast_spacing, PT_UCHAR, 0, 0);
    if (ctm_state->uchar_vol == 0) {
	print_and_exit ("ERROR: failed in allocating the volume");
    }

    /* Create output volume for labelmap */
    ctm_state->labelmap_vol = 0;
    if (want_labelmap) {
	ctm_state->labelmap_vol = volume_create (cxt->rast_dim, 
	    cxt->rast_offset, cxt->rast_spacing, PT_UINT32, 0, 0);
	if (ctm_state->labelmap_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Create output volume for ss_img */
    ctm_state->ss_img_vol = 0;
    if (want_ss_img) {
	ctm_state->ss_img_vol = volume_create (cxt->rast_dim, 
	    cxt->rast_offset, cxt->rast_spacing, PT_UINT32, 0, 0);
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
	return false;
    }
    
    curr_structure = cxt->slist[ctm_state->curr_struct_no];
    slice_voxels = cxt->rast_dim[0] * cxt->rast_dim[1];

    memset (uchar_img, 0, cxt->rast_dim[0] * cxt->rast_dim[1] 
		* cxt->rast_dim[2] * sizeof(unsigned char));

    /* Loop through polylines in this structure */
    for (int i = 0; i < curr_structure->num_contours; i++) {
	Rtss_polyline* curr_contour;
	unsigned char* uchar_slice;
	int slice_no;

	curr_contour = curr_structure->pslist[i];
	if (curr_contour->num_vertices == 0) {
	    continue;
	}
	slice_no = ROUND_INT((curr_contour->z[0] - cxt->rast_offset[2]) 
	    / cxt->rast_spacing[2]);
	if (slice_no < 0 || slice_no >= cxt->rast_dim[2]) {
	    continue;
	}

	/* Render contour to binary */
	memset (ctm_state->acc_img, 0, slice_voxels * sizeof(unsigned char));
	render_slice_polyline (
	    ctm_state->acc_img, 
	    cxt->rast_dim, 
	    cxt->rast_spacing, 
	    cxt->rast_offset,
	    curr_contour->num_vertices, 
	    curr_contour->x, 
	    curr_contour->y);

	/* Copy from acc_img into mask image */
	if (ctm_state->want_prefix_imgs) {
	    uchar_slice = &uchar_img[slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		uchar_slice[k] ^= ctm_state->acc_img[k];
	    }
	}

	/* Copy from acc_img into labelmask and xormap images */
	if (ctm_state->want_labelmap) {
	    uint32_t* labelmap_img;
	    uint32_t* uint32_slice;
	    labelmap_img = (uint32_t*) ctm_state->labelmap_vol->img;
	    uint32_slice = &labelmap_img[slice_no * slice_voxels];
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
	    uint32_slice = &ss_img_img[slice_no * slice_voxels];
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
