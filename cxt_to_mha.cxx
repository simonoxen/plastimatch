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
#include "rtss_polyline_set.h"
#include "rtss_structure.h"
#include "volume.h"

void
cxt_to_mha_init (
    Cxt_to_mha_state *ctm_state,       /* Output */
    Rtss_polyline_set *cxt,            /* Input */
    Plm_image_header *pih,             /* Input */
    bool want_prefix_imgs,             /* Input */
    bool want_labelmap,                /* Input */
    bool want_ss_img                   /* Input */
)
{
    int slice_voxels;

    pih->get_origin (ctm_state->origin);
    pih->get_spacing (ctm_state->spacing);
    pih->get_dim (ctm_state->dim);

    slice_voxels = ctm_state->dim[0] * ctm_state->dim[1];

    ctm_state->want_prefix_imgs = want_prefix_imgs;
    ctm_state->want_labelmap = want_labelmap;
    ctm_state->want_ss_img = want_ss_img;

    ctm_state->acc_img = (unsigned char*) malloc (
	slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
       structure */
    ctm_state->uchar_vol = volume_create (ctm_state->dim, ctm_state->origin, 
	ctm_state->spacing, PT_UCHAR, 0, 0);
    if (ctm_state->uchar_vol == 0) {
	print_and_exit ("ERROR: failed in allocating the volume");
    }

    /* Create output volume for labelmap */
    ctm_state->labelmap_vol = 0;
    if (want_labelmap) {
	ctm_state->labelmap_vol = volume_create (ctm_state->dim, 
	    ctm_state->origin, ctm_state->spacing, PT_UINT32, 0, 0);
	if (ctm_state->labelmap_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Create output volume for ss_img */
#if (PLM_USE_SS_IMAGE_VEC)
    if (want_ss_img) {
	ctm_state->m_ss_img = UCharVecImageType::New ();
	itk_image_set_header (ctm_state->m_ss_img, pih);
	int num_uchar = 1 + (cxt->num_structures-1) / 8;
	if (num_uchar < 2) num_uchar = 2;
	ctm_state->m_ss_img->SetVectorLength (num_uchar);
	ctm_state->m_ss_img->Allocate ();
    }
#else
    ctm_state->ss_img_vol = 0;
    if (want_ss_img) {
	ctm_state->ss_img_vol = volume_create (ctm_state->dim, 
	    ctm_state->origin, ctm_state->spacing, PT_UINT32, 0, 0);
	if (ctm_state->ss_img_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }
#endif

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
    slice_voxels = ctm_state->dim[0] * ctm_state->dim[1];

    memset (uchar_img, 0, ctm_state->dim[0] * ctm_state->dim[1] 
	* ctm_state->dim[2] * sizeof(unsigned char));

    /* Loop through polylines in this structure */
    for (int i = 0; i < curr_structure->num_contours; i++) {
	Rtss_polyline* curr_contour;
	unsigned char* uchar_slice;
	int slice_no;

	curr_contour = curr_structure->pslist[i];
	if (curr_contour->num_vertices == 0) {
	    continue;
	}
	slice_no = ROUND_INT((curr_contour->z[0] - ctm_state->origin[2]) 
	    / ctm_state->spacing[2]);
	if (slice_no < 0 || slice_no >= ctm_state->dim[2]) {
	    continue;
	}

	/* Render contour to binary */
	memset (ctm_state->acc_img, 0, slice_voxels * sizeof(unsigned char));
	render_slice_polyline (
	    ctm_state->acc_img, 
	    ctm_state->dim, 
	    ctm_state->spacing, 
	    ctm_state->origin,
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
#if (PLM_USE_SS_IMAGE_VEC)
	    /* GCS FIX: This code is replicated in ss_img_extract */
	    unsigned int uchar_no = ctm_state->curr_bit / 8;
	    unsigned int bit_no = ctm_state->curr_bit % 8;
	    unsigned char bit_mask = 1 << bit_no;
	    printf ("Computed bit mask %d -> bit (%d,%d) 0x%02x\n", 
		ctm_state->curr_bit, uchar_no, bit_no, bit_mask);
	    if (uchar_no > ctm_state->m_ss_img->GetVectorLength()) {
		print_and_exit (
		    "Error: bit %d was requested from image of %d bits\n", 
		    ctm_state->curr_bit, 
		    ctm_state->m_ss_img->GetVectorLength() * 8);
	    }
	    /* GCS FIX: This is inefficient, due to undesirable construct 
	       and destruct of itk::VariableLengthVector of each pixel */
	    UCharVecImageType::IndexType idx = {{0, 0, slice_no}};
	    for (idx.m_Index[1] = 0; 
		 idx.m_Index[1] < ctm_state->dim[1]; 
		 idx.m_Index[1]++) {
		for (idx.m_Index[0] = 0; 
		     idx.m_Index[0] < ctm_state->dim[0]; 
		     idx.m_Index[0]++) {
		    itk::VariableLengthVector<unsigned char> v 
			= ctm_state->m_ss_img->GetPixel (idx);
		    v[uchar_no] |= (1 << ctm_state->curr_bit);
		    ctm_state->m_ss_img->SetPixel (idx, v);
		}		
	    }
#else
	    uint32_t* ss_img_img = 0;
	    uint32_t* uint32_slice;
	    ss_img_img = (uint32_t*) ctm_state->ss_img_vol->img;
	    uint32_slice = &ss_img_img[slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		if (ctm_state->acc_img[k]) {
		    uint32_slice[k] |= (1 << ctm_state->curr_bit);
		}
	    }
#endif
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
#if (!PLM_USE_SS_IMAGE_VEC)
    if (ctm_state->ss_img_vol) {
	volume_destroy (ctm_state->ss_img_vol);
    }
#endif
    free (ctm_state->acc_img);
}

Cxt_to_mha_state*
cxt_to_mha_create (
    Rtss_polyline_set *cxt,
    Plm_image_header *pih
)
{
    Cxt_to_mha_state *ctm_state = new Cxt_to_mha_state;
    cxt_to_mha_init (ctm_state, cxt, pih, false, true, true);
    while (cxt_to_mha_process_next (ctm_state, cxt)) {}

    return ctm_state;
}

void
cxt_to_mha_destroy (Cxt_to_mha_state *ctm_state)
{
    cxt_to_mha_free (ctm_state);
    delete ctm_state;
}
