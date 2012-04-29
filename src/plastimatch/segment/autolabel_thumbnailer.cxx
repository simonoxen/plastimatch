/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImageRegionIterator.h"

#include "autolabel_thumbnailer.h"
#include "dlib_trainer.h"
#include "itk_image.h"
#include "thumbnail.h"

Autolabel_thumbnailer::Autolabel_thumbnailer ()
{
    pli = 0;
    thumb = new Thumbnail;
}

Autolabel_thumbnailer::~Autolabel_thumbnailer ()
{
    if (pli) delete pli;
    if (thumb) delete thumb;
}

void
Autolabel_thumbnailer::set_input_image (const char* fn)
{
    if (pli) delete pli;
    pli = plm_image_load (fn, PLM_IMG_TYPE_ITK_FLOAT);
    thumb->set_input_image (pli);
    thumb->set_thumbnail_dim (16);
    thumb->set_thumbnail_spacing (25.0f);
}

Dlib_trainer::Dense_sample_type 
Autolabel_thumbnailer::make_sample (float slice_loc)
{
    thumb->set_slice_loc (slice_loc);
    FloatImageType::Pointer thumb_img = thumb->make_thumbnail ();

    itk::ImageRegionIterator< FloatImageType > thumb_it (
        thumb_img, thumb_img->GetLargestPossibleRegion());
    Dlib_trainer::Dense_sample_type d;
    for (int j = 0; j < 256; j++) {
        d(j) = thumb_it.Get();
        ++thumb_it;
    }
    return d;
}
