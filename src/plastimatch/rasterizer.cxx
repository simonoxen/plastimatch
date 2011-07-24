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
#include "file_util.h"
#include "math_util.h"
#include "mha_io.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "plm_path.h"
#include "rasterizer.h"
#include "rasterize_slice.h"
#include "rtss_polyline_set.h"
#include "rtss_structure.h"
#include "volume.h"

void
Rasterizer::init (
    Rtss_polyline_set *cxt,            /* Input */
    Plm_image_header *pih,             /* Input */
    bool want_prefix_imgs,             /* Input */
    bool want_labelmap,                /* Input */
    bool want_ss_img                   /* Input */
)
{
    int slice_voxels;

    pih->get_origin (this->origin);
    pih->get_spacing (this->spacing);
    pih->get_dim (this->dim);

    slice_voxels = this->dim[0] * this->dim[1];

    this->want_prefix_imgs = want_prefix_imgs;
    this->want_labelmap = want_labelmap;
    this->want_ss_img = want_ss_img;

    this->acc_img = (unsigned char*) malloc (
	slice_voxels * sizeof(unsigned char));

    /* Create output volume for mask image.  This is reused for each 
       structure */
    this->uchar_vol = new Volume (this->dim, this->origin, 
	this->spacing, 0, PT_UCHAR, 1, 0);
    if (this->uchar_vol == 0) {
	print_and_exit ("ERROR: failed in allocating the volume");
    }

    /* Create output volume for labelmap */
    this->labelmap_vol = 0;
    if (want_labelmap) {
	this->labelmap_vol = new Volume (this->dim, 
	    this->origin, this->spacing, 0, PT_UINT32, 1, 0);
	if (this->labelmap_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }

    /* Create output volume for ss_img */
#if (PLM_USE_SS_IMAGE_VEC)
    if (want_ss_img) {
	this->m_ss_img = UCharVecImageType::New ();
	itk_image_set_header (this->m_ss_img, pih);
	int num_uchar = 1 + (cxt->num_structures-1) / 8;
	if (num_uchar < 2) num_uchar = 2;
	this->m_ss_img->SetVectorLength (num_uchar);
	this->m_ss_img->Allocate ();
    }
#else
    this->ss_img_vol = 0;
    if (want_ss_img) {
	this->ss_img_vol = new Volume (this->dim, 
	    this->origin, this->spacing, 0, PT_UINT32, 1, 0);
	if (this->ss_img_vol == 0) {
	    print_and_exit ("ERROR: failed in allocating the volume");
	}
    }
#endif

    /* Initialize to start with first structure */
    this->curr_struct_no = 0;
    this->curr_bit = 0;
}

/* Return true if an image was processed */
bool
Rasterizer::process_next (
    Rtss_polyline_set *cxt                          /* In/out */
)
{
    Rtss_structure* curr_structure;
    unsigned char* uchar_img = (unsigned char*) this->uchar_vol->img;
    int slice_voxels;

    /* If done, return false */
    if (this->curr_struct_no >= cxt->num_structures) {
	this->curr_struct_no = cxt->num_structures + 1;
	return false;
    }
    
    curr_structure = cxt->slist[this->curr_struct_no];
    slice_voxels = this->dim[0] * this->dim[1];

    memset (uchar_img, 0, this->dim[0] * this->dim[1] 
	* this->dim[2] * sizeof(unsigned char));

    /* Loop through polylines in this structure */
    for (int i = 0; i < curr_structure->num_contours; i++) {
	Rtss_polyline* curr_contour;
	unsigned char* uchar_slice;
	int slice_no;

	curr_contour = curr_structure->pslist[i];
	if (curr_contour->num_vertices == 0) {
	    continue;
	}
	slice_no = ROUND_INT((curr_contour->z[0] - this->origin[2]) 
	    / this->spacing[2]);
	if (slice_no < 0 || slice_no >= this->dim[2]) {
	    continue;
	}

	/* Render contour to binary */
	memset (this->acc_img, 0, slice_voxels * sizeof(unsigned char));
	rasterize_slice (
	    this->acc_img, 
	    this->dim, 
	    this->spacing, 
	    this->origin,
	    curr_contour->num_vertices, 
	    curr_contour->x, 
	    curr_contour->y);

	/* Copy from acc_img into mask image */
	if (this->want_prefix_imgs) {
	    uchar_slice = &uchar_img[slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		uchar_slice[k] ^= this->acc_img[k];
	    }
	}

	/* Copy from acc_img into labelmask and xormap images */
	if (this->want_labelmap) {
	    uint32_t* labelmap_img;
	    uint32_t* uint32_slice;
	    labelmap_img = (uint32_t*) this->labelmap_vol->img;
	    uint32_slice = &labelmap_img[slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		if (this->acc_img[k]) {
		    uint32_slice[k] = this->curr_bit + 1;
		}
	    }
	}
	if (this->want_ss_img) {
#if (PLM_USE_SS_IMAGE_VEC)
	    /* GCS FIX: This code is replicated in ss_img_extract */
	    unsigned int uchar_no = this->curr_bit / 8;
	    unsigned int bit_no = this->curr_bit % 8;
	    unsigned char bit_mask = 1 << bit_no;
	    if (uchar_no > this->m_ss_img->GetVectorLength()) {
		print_and_exit (
		    "Error: bit %d was requested from image of %d bits\n", 
		    this->curr_bit, 
		    this->m_ss_img->GetVectorLength() * 8);
	    }
	    /* GCS FIX: This is inefficient, due to undesirable construct 
	       and destruct of itk::VariableLengthVector of each pixel */
	    UCharVecImageType::IndexType idx = {{0, 0, slice_no}};
	    int k = 0;
	    for (idx.m_Index[1] = 0; 
		 idx.m_Index[1] < this->dim[1]; 
		 idx.m_Index[1]++) {
		for (idx.m_Index[0] = 0; 
		     idx.m_Index[0] < this->dim[0]; 
		     idx.m_Index[0]++) {
		    if (this->acc_img[k]) {
			itk::VariableLengthVector<unsigned char> v 
			    = this->m_ss_img->GetPixel (idx);
			v[uchar_no] |= bit_mask;
			this->m_ss_img->SetPixel (idx, v);
		    }
		    k++;
		}
	    }
#else
	    uint32_t* ss_img_img = 0;
	    uint32_t* uint32_slice;
	    ss_img_img = (uint32_t*) this->ss_img_vol->img;
	    uint32_slice = &ss_img_img[slice_no * slice_voxels];
	    for (int k = 0; k < slice_voxels; k++) {
		if (this->acc_img[k]) {
		    uint32_slice[k] |= (1 << this->curr_bit);
		}
	    }
#endif
	}
    }

    this->curr_struct_no ++;
    if (curr_structure->num_contours > 0) {
	curr_structure->bit = this->curr_bit;
	this->curr_bit ++;
    }

    return true;
}

const char*
Rasterizer::current_name (
    Rtss_polyline_set *cxt
)
{
    if (this->curr_struct_no < cxt->num_structures + 1) {
	Rtss_structure *curr_structure;
	curr_structure = cxt->slist[this->curr_struct_no-1];
	return curr_structure->name;
    } else {
	return "";
    }
}

Rasterizer::~Rasterizer (void)
{
    if (this->uchar_vol) {
	delete this->uchar_vol;
    }
    if (this->labelmap_vol) {
	delete this->labelmap_vol;
    }
#if (!PLM_USE_SS_IMAGE_VEC)
    if (this->ss_img_vol) {
	delete this->ss_img_vol;
    }
#endif
    free (this->acc_img);
}

void
Rasterizer::rasterize (
    Rtss_polyline_set *cxt,            /* Input */
    Plm_image_header *pih,             /* Input */
    bool want_prefix_imgs,             /* Input */
    bool want_labelmap,                /* Input */
    bool want_ss_img                   /* Input */
)
{
    this->init (cxt, pih, false, true, true);
    while (this->process_next (cxt)) {}
}
