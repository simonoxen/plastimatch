/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "plmbase.h"
#include "plmsys.h"


Thumbnail::Thumbnail ()
{
    pli = 0;
    axis = 2;
    thumbnail_dim = 16;
    thumbnail_spacing = 30.0;
    center[0] = center[1] = center[2] = 0;
    slice_loc = 0;
}

void 
Thumbnail::set_internal_geometry ()
{
    for (int d = 0; d < 3; d++) {
	origin[d] = center[d] 
	    - thumbnail_spacing * (thumbnail_dim - 1) / 2;
	spacing[d] = thumbnail_spacing;
	dim[d] = thumbnail_dim;
    }
    origin[axis] = slice_loc;
    spacing[axis] = 1;
    dim[axis] = 1;
}

void 
Thumbnail::set_input_image (Plm_image *pli)
{
    Plm_image_header pih;

    this->pli = pli;

/* This should be explicit request by caller */
#if defined (commentout)
    pih.get_image_center (this->center);
    if (!slice_loc_was_set) {
	slice_loc = center[axis];
    }
#endif
}

void 
Thumbnail::set_slice_loc (float slice_loc)
{
    this->slice_loc = slice_loc;
}

void 
Thumbnail::set_axis (int axis)
{
    if (axis < 0 || axis > 2) {
        print_and_exit ("Error, thumbnail axis must be between 0 and 2\n");
    }
    this->axis = axis;
}

void 
Thumbnail::set_thumbnail_dim (int thumb_dim)
{
    this->thumbnail_dim = thumb_dim;
}

void 
Thumbnail::set_thumbnail_spacing (float thumb_spacing)
{
    this->thumbnail_spacing = thumb_spacing;
}

FloatImageType::Pointer 
Thumbnail::make_thumbnail ()
{
    /* Figure out resampling geometry */
    set_internal_geometry ();

    /* Resample the image */
    Plm_image_header pih (dim, origin, spacing);
    FloatImageType::Pointer itk_resampled_image = 
        resample_image (pli->m_itk_float, &pih, -1000, 1);
    Plm_image plm_resampled_image (itk_resampled_image);

    /* Reshuffle dimensions to 2D */
    if (axis == 0) {
        Volume *vol = plm_resampled_image.gpuit_float();
        vol->dim[0] = vol->dim[1];
        vol->dim[1] = vol->dim[2];
        vol->dim[2] = 1;
        /* GCS FIX: Do something about spacing here */
        /* GCS FIX: Do something about direction cosines here */
    }
    else if (axis == 1) {
        Volume *vol = plm_resampled_image.gpuit_float();
        vol->dim[1] = vol->dim[2];
        vol->dim[2] = 1;
        /* GCS FIX: Do something about spacing here */
        /* GCS FIX: Do something about direction cosines here */
    }
    else {
        /* Do nothing */
    }

    return plm_resampled_image.itk_float ();
}
