/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "plm_registration.h"
#include "plm_image.h"
#include "itk_registration.h"
#include "itk_optim.h"
#include "resample_mha.h"
#include "itk_warp.h"
#include "itk_demons.h"
#include "gpuit_bspline.h"
#include "gpuit_demons.h"
#include "xform.h"
#include "readmha.h"

#define FIXME_BACKGROUND_MAX (-1200)

/* This helps speed up the registration, by setting the bounding box to the 
   smallest size needed.  To find the bounding box, either use the extent 
   of the fixed_mask (if one is used), or by eliminating excess air by thresholding
*/
static void
set_fixed_image_region_global (Registration_Data* regd)
{
    int use_magic_value = 1;

    regd->fixed_region_origin = regd->fixed_image->itk_float()->GetOrigin();
    regd->fixed_region_spacing = regd->fixed_image->itk_float()->GetSpacing();

    if (regd->fixed_mask) {
	FloatImageType::RegionType::IndexType valid_index;
	FloatImageType::RegionType::SizeType valid_size;

	/* Search for bounding box of fixed mask */
	typedef itk::ImageRegionConstIteratorWithIndex< UCharImageType > IteratorType;
	UCharImageType::RegionType region = regd->fixed_mask->GetLargestPossibleRegion();
	IteratorType it (regd->fixed_mask, region);

	int first = 1;
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    unsigned char c = it.Get();
	    if (c) {
		UCharImageType::RegionType::IndexType idx = it.GetIndex();
		if (first) {
		    first = 0;
		    valid_index = idx;
		    valid_size[0] = 1;
		    valid_size[1] = 1;
		    valid_size[2] = 1;
		} else {
		    int updated = 0;
		    for (int i = 0; i < 3; i++) {
			if (valid_index[i] > idx[i]) {
			    valid_size[i] += valid_index[i] - idx[i];
			    valid_index[i] = idx[i];
			    updated = 1;
			}
			if (idx[i] - valid_index[i] >= valid_size[i]) {
			    valid_size[i] = idx[i] - valid_index[i] + 1;
			    updated = 1;
			}
		    }
		}
	    }
	}
	regd->fixed_region.SetIndex(valid_index);
	regd->fixed_region.SetSize(valid_size);
    } else if (use_magic_value) {
	FloatImageType::RegionType::IndexType valid_index;
	FloatImageType::RegionType::SizeType valid_size;

	/* Make sure the image is ITK float */
	FloatImageType::Pointer fixed_image = regd->fixed_image->itk_float();

	/* Search for bounding box of patient */
	typedef itk::ImageRegionConstIteratorWithIndex<FloatImageType> IteratorType;
	FloatImageType::RegionType region = fixed_image->GetLargestPossibleRegion();
	IteratorType it (fixed_image, region);

	int first = 1;
	for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	    float c = it.Get();
	    if (c > FIXME_BACKGROUND_MAX) {
		FloatImageType::RegionType::IndexType idx = it.GetIndex();
		if (first) {
		    first = 0;
		    valid_index = idx;
		    valid_size[0] = 1;
		    valid_size[1] = 1;
		    valid_size[2] = 1;
		} else {
		    int updated = 0;
		    for (int i = 0; i < 3; i++) {
			if (valid_index[i] > idx[i]) {
			    valid_size[i] += valid_index[i] - idx[i];
			    valid_index[i] = idx[i];
			    updated = 1;
			}
			if (idx[i] - valid_index[i] >= valid_size[i]) {
			    valid_size[i] = idx[i] - valid_index[i] + 1;
			    updated = 1;
			}
		    }
		}
	    }
	}
	/* Try to include a margin of at least one air pixel everywhere */
	for (int i = 0; i < 3; i++) {
	    if (valid_index[i] > 0) {
		valid_index[i]--;
		valid_size[i]++;
	    }
	    if (valid_size[i] + valid_index[i] < fixed_image->GetLargestPossibleRegion().GetSize()[i]) {
		valid_size[i]++;
	    }
	}
	regd->fixed_region.SetIndex(valid_index);
	regd->fixed_region.SetSize(valid_size);
    } else {
	regd->fixed_region = regd->fixed_image->itk_float()->GetLargestPossibleRegion();
    }
}

static PlmImageType
choose_image_type (int xform_type, int optim_type, int impl_type)
{
    switch (impl_type) {
	case IMPLEMENTATION_GPUIT_CPU:
	case IMPLEMENTATION_GPUIT_BROOK:
	    return PLM_IMG_TYPE_GPUIT_FLOAT;
	default:
	    return PLM_IMG_TYPE_ITK_FLOAT;
    }
}

void
save_warped_img_itk (Registration_Data* regd,
		 DeformationFieldType::Pointer vf,
		 int img_out_fmt, 
		 char* fn)
{
    FloatImageType::Pointer im_warped = FloatImageType::New();

    printf ("Converting image 1...\n");
    FloatImageType::Pointer i1 = regd->moving_image->itk_float();
    printf ("Converting image 2...\n");
    FloatImageType::Pointer i2 = regd->fixed_image->itk_float();

    printf ("Warping image...\n");
    im_warped = itk_warp_image (regd->moving_image->itk_float(), vf, 1, 0.0f);
    printf ("Saving image...\n");
    if (img_out_fmt == IMG_OUT_FMT_AUTO) {
	save_short (im_warped, fn);
    } else if (img_out_fmt == IMG_OUT_FMT_DICOM) {
	save_short_dicom (im_warped, fn);
    } else {
	print_and_exit ("Program error.  Unknown output file type.\n");
    }
}

void
save_warped_img_gpuit (Registration_Data* regd,
			Volume* vf, char* fn)
{
    Volume *vin, *vout;

    printf ("Converting image itk->gpuit (was %d)\n",
	    regd->moving_image->m_type);
    vin = regd->moving_image->gpuit_float();

    printf ("Warping image...\n");

    vout = volume_create (vf->dim, vf->offset, vf->pix_spacing, PT_FLOAT, 0);
    volume_warp (vout, vin, vf);

    printf ("Output image: npix = %d\n", vout->npix);

    printf ("Saving image...\n");
    //volume_convert_to_short (vout);
    write_mha (fn, vout);

    //write_mha (fn, vin);

    volume_free (vout);
}

void
save_stage_output (Registration_Data* regd, Xform *xf_out, Stage_Parms* stage)
{
    Xform xf_tmp;

    if (stage->img_out_fn[0] || stage->vf_out_fn[0]) {
	/* Convert xform to vf */
	printf ("Converting xf to vector field ...\n");
	xform_to_itk_vf (&xf_tmp, xf_out, regd->fixed_image->itk_float());
	/* Save warped image */
	if (stage->img_out_fn[0]) {
	    printf ("Saving warped image ...\n");
	    save_warped_img_itk (regd, xf_tmp.get_itk_vf(), stage->img_out_fmt, 
		    stage->img_out_fn);
	}
	/* Save deformation field */
	if (stage->vf_out_fn[0]) {
	    printf ("Writing vector field ...\n");
	    save_image (xf_tmp.get_itk_vf(), stage->vf_out_fn);
	}
    }

    if (stage->xf_out_fn[0]) {
	printf ("Writing deformation parameters ...\n");
	save_xform (xf_out, stage->xf_out_fn);
    }
}

void
save_regp_output_gpuit (Registration_Data* regd, Xform *xf_out, Registration_Parms* regp)
{
    if (regp->xf_out_fn[0]) {
	printf ("Writing transformation ...\n");
	save_xform (xf_out, regp->xf_out_fn);
    }

    if (regp->img_out_fn[0] || regp->vf_out_fn[0]) {
	Xform xf_gpuit_vf;
	int d;
	int dim[3];
	float offset[3];
	float spacing[3];

	/* Convert xform to vf */
	printf ("Converting xf to vector field ...\n");

	const FloatImageType::Pointer fixed = regd->fixed_image->itk_float();
	FloatImageType::SizeType img_sz = fixed->GetLargestPossibleRegion().GetSize();
	FloatImageType::PointType img_og = fixed->GetOrigin();
	FloatImageType::SpacingType img_sp = fixed->GetSpacing();

	/* Copy header & allocate data for itk */
	for (d = 0; d < 3; d++) {
	    dim[d] = img_sz[d];
	    offset[d] = img_og[d];
	    spacing[d] = img_sp[d];
	}
	xform_to_gpuit_vf (&xf_gpuit_vf, xf_out, dim, offset, spacing);

	vf_print_stats (xf_gpuit_vf.get_gpuit_vf());

	/* Save deformation field */
	if (regp->vf_out_fn[0]) {
	    Volume* vf = xf_gpuit_vf.get_gpuit_vf();
	    printf ("Writing vector field ...\n");
	    write_mha (regp->vf_out_fn, vf);
	}

	/* Save warped image */
	if (regp->img_out_fn[0]) {
	    save_warped_img_gpuit (regd, xf_gpuit_vf.get_gpuit_vf(), regp->img_out_fn);
	}
    }
}

void
save_regp_output_itk (Registration_Data* regd, Xform *xf_out, Registration_Parms* regp)
{
    Xform xf_tmp;

#if defined (commentout)
void
itk_bsp_extend_to_region (Xform* xf,
		      const OriginType& img_origin,
		      const SpacingType& img_spacing,
		      const ImageRegionType& img_region);
#endif

    if (regp->xf_out_fn[0]) {
	printf ("Writing transformation ...\n");
	save_xform (xf_out, regp->xf_out_fn);
    }

    /* GCS DEBUGGING... */
#if defined (commentout)
    printf ("Trying to extend region...\n");
    itk_bsp_extend_to_region (xf_out, 
				regd->fixed_image->itk_float()->GetOrigin(), 
				regd->fixed_image->itk_float()->GetSpacing(), 
				regd->fixed_image->itk_float()->GetLargestPossibleRegion());
    save_xform (xf_out, "extended.txt");
#endif

    if (regp->img_out_fn[0] || regp->vf_out_fn[0]) {

	/* Convert xform to vf */
	printf ("Converting xf to vector field ...\n");
	const FloatImageType::SizeType& img_size = regd->fixed_image->itk_float()->GetLargestPossibleRegion().GetSize();
	xform_to_itk_vf (&xf_tmp, xf_out, regd->fixed_image->itk_float());

	/* Save warped image */
	if (regp->img_out_fn[0]) {
	    printf ("Saving warped image ...\n");
	    save_warped_img_itk (regd, xf_tmp.get_itk_vf(), regp->img_out_fmt, 
		    regp->img_out_fn);
	}
	/* Save deformation field */
	if (regp->vf_out_fn[0]) {
	    printf ("Writing vector field ...\n");

#define USE_BUGGY_ITK 1
#if defined (USE_BUGGY_ITK)
	    save_image (xf_tmp.get_itk_vf(), regp->vf_out_fn);
#else
	    {
		Xform xf2;
		int d;
		int dim[3];
		float offset[3];
		float spacing[3];

	        DeformationFieldType::Pointer itk_vf = xf_tmp.get_itk_vf();
		DeformationFieldType::RegionType rg = itk_vf->GetLargestPossibleRegion();
		DeformationFieldType::PointType og = itk_vf->GetOrigin();
		DeformationFieldType::SpacingType sp = itk_vf->GetSpacing();
		DeformationFieldType::SizeType sz = rg.GetSize();

	        /* Copy header & allocate data for itk */
		for (d = 0; d < 3; d++) {
		    dim[d] = sz[d];
		    offset[d] = og[d];
		    spacing[d] = sp[d];
		}
		xform_to_gpuit_vf (&xf2, &xf_tmp, dim, offset, spacing);

		Volume* vf = xf2.get_gpuit_vf();
		write_mha (regp->vf_out_fn, vf);
	    }
#endif
	}
    }
}

void
do_registration_stage (Registration_Data* regd, Xform *xf_out, Xform *xf_in, 
		       Stage_Parms* stage)
{
    /* Convert image types */
    PlmImageType image_type = choose_image_type (stage->xform_type, stage->optim_type, stage->impl_type);

    printf ("xf_in->m_type = %d, xf_out->m_type = %d\n", xf_in->m_type, xf_out->m_type);

    /* Run registration */
    if (stage->optim_type == OPTIMIZATION_DEMONS) {
	if (stage->impl_type == IMPLEMENTATION_ITK) {
	    do_demons_stage (regd, xf_out, xf_in, stage);
	} else {
	    do_gpuit_demons_stage (regd, xf_out, xf_in, stage);
	}
    }
    else if (stage->xform_type == STAGE_TRANSFORM_BSPLINE) {
	if (stage->impl_type == IMPLEMENTATION_ITK) {
	    do_itk_stage (regd, xf_out, xf_in, stage);
	} else {
	    do_gpuit_bspline_stage (regd, xf_out, xf_in, stage);
	}
    }
    else {
	do_itk_stage (regd, xf_out, xf_in, stage);
    }

    printf ("xf_out->m_type = %d, xf_in->m_type = %d\n", xf_out->m_type, xf_in->m_type);

    /* Save intermediate output */
    save_stage_output (regd, xf_out, stage);
}

void
load_input_files (Registration_Data* regd, Registration_Parms* regp)
{
    PlmImageType image_type = PLM_IMG_TYPE_ITK_FLOAT;

#if defined (commentout)
    /* Load the appropriate image type for the first stage */
    if (regp->num_stages > 0) {
	image_type = choose_image_type (regp->stages[0]->xform_type,
	    regp->stages[0]->optim_type, regp->stages[0]->impl_type);
    }
#endif

    /* GCS Jun 2, 2008.  Always load as ITK so we can find the ROI */

    printf ("fixed image=%s\n", regp->fixed_fn);
    printf ("Loading fixed image...");
    //regd->fixed_image = load_float (regp->fixed_fn);
    regd->fixed_image = rad_image_load (regp->fixed_fn, image_type);
    fflush (stdout);
    printf ("done!\n");

    printf ("moving image=%s\n", regp->moving_fn);
    printf ("Loading moving image...");
    fflush (stdout);
    regd->moving_image = rad_image_load (regp->moving_fn, image_type);
    printf ("done!\n");

    if (regp->fixed_mask_fn[0]) {
	printf ("Loading fixed mask...");
	regd->fixed_mask = load_uchar (regp->fixed_mask_fn);
	printf ("done!\n");
    } else {
	regd->fixed_mask = 0;
    }
    if (regp->moving_mask_fn[0]) {
	printf ("Loading moving mask...");
	regd->moving_mask = load_uchar (regp->moving_mask_fn);
	printf ("done!\n");
    } else {
	regd->moving_mask = 0;
    }
}

void
do_registration (Registration_Parms* regp)
{
    int i;
    Registration_Data regd;
    Xform xf1, xf2;
    Xform *xf_in, *xf_out, *xf_tmp;

    xf_in = &xf1;
    xf_out = &xf2;

    /* Load images */
    load_input_files (&regd, regp);

    /* Load initial guess of xform */
    if (regp->xf_in_fn[0]) {
	load_xform (xf_out, regp->xf_in_fn);
    }

    /* Set fixed image region */
    set_fixed_image_region_global (&regd);

    for (i = 0; i < regp->num_stages; i++) {
	/* Swap xf_in and xf_out */
	xf_tmp = xf_out; xf_out = xf_in; xf_in = xf_tmp;
	/* Run registation, results are stored in xf_out */
	do_registration_stage (&regd, xf_out, xf_in, regp->stages[i]);
    }

    /* RMK: If no stages, we still generate output (same as input) */

    save_regp_output_itk (&regd, xf_out, regp);
}
