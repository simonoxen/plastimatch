/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "resample_mha.h"
#include "thumbnail.h"

Thumbnail::Thumbnail ()
{
    pli = 0;
    thumbnail_dim = 16;
    thumbnail_spacing = 30.0;
    center[0] = center[1] = center[2] = 0;
    slice_loc = 0;
}

void 
Thumbnail::set_internal_geometry ()
{
    for (int d = 0; d < 2; d++) {
	origin[d] = center[d] 
	    - thumbnail_spacing * (thumbnail_dim - 1) / 2;
	spacing[d] = thumbnail_spacing;
	dim[d] = thumbnail_dim;
    }
    origin[2] = slice_loc;
    spacing[2] = 1;
    dim[2] = 1;
}

void 
Thumbnail::set_input_image (Plm_image *pli)
{
    Plm_image_header pih;

    this->pli = pli;
    pih.get_image_center (this->center);
    if (!slice_loc_was_set) {
	slice_loc = center[2];
    }
    set_internal_geometry ();
}

void 
Thumbnail::set_slice_loc (float slice_loc)
{
    this->slice_loc = slice_loc;
    this->origin[2] = slice_loc;
}

void 
Thumbnail::set_thumbnail_dim (int thumb_dim)
{
    this->thumbnail_dim = thumb_dim;
    set_internal_geometry ();
}

void 
Thumbnail::set_thumbnail_spacing (float thumb_spacing)
{
    this->thumbnail_spacing = thumb_spacing;
    set_internal_geometry ();
}

FloatImageType::Pointer 
Thumbnail::make_thumbnail ()
{
    /* Resample the image */
    return resample_image (pli->m_itk_float, origin, spacing, dim, -1000, 1);
}
