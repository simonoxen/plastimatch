/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "rad_image.h"
#include "itkImageRegionIterator.h"
#include "readmha.h"
#include "volume.h"
#include "print_and_exit.h"

RadImage*
rad_image_load (char* fname, RadImage::RadImageType type)
{
    RadImage *ri = new RadImage;
    if (!ri) return 0;

    switch (type) {
	case RadImage::TYPE_GPUIT_FLOAT:
	    ri->m_type = type;
	    ri->m_gpuit = read_mha (fname);
	    break;
	case RadImage::TYPE_ITK_FLOAT:
	    ri->m_type = type;
	    ri->m_itk_float = load_float (fname);
	    break;
	default:
	    print_and_exit ("Unhandled image load in rad_image_load\n");
	    break;
    }
    return ri;
}

void
RadImage::convert_itk_float ()
{
    switch (this->m_type) {
    case RadImage::TYPE_ITK_FLOAT:
	return;
    case RadImage::TYPE_GPUIT_FLOAT:
	{
	    int i;
	    Volume* vol = (Volume*) m_gpuit;
	    float* img = (float*) vol->img;
	    FloatImageType::SizeType sz;
	    FloatImageType::IndexType st;
	    FloatImageType::RegionType rg;
	    FloatImageType::PointType og;
	    FloatImageType::SpacingType sp;

	    /* Copy header & allocate data for itk */
	    for (i = 0; i < 3; i++) {
		st[i] = 0;
		sz[i] = vol->dim[i];
		sp[i] = vol->pix_spacing[i];
		og[i] = vol->offset[i];
	    }
	    rg.SetSize (sz);
	    rg.SetIndex (st);

	    this->m_itk_float = FloatImageType::New();
	    this->m_itk_float->SetRegions (rg);
	    this->m_itk_float->SetOrigin (og);
	    this->m_itk_float->SetSpacing (sp);
	    this->m_itk_float->Allocate();

	    /* Copy data into itk */
	    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
	    FloatIteratorType it (this->m_itk_float, rg);
	    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
		it.Set (img[i]);
	    }

	    /* Free gpuit data */
	    volume_free (vol);
	    this->m_gpuit = 0;

	    /* Set data type */
	    this->m_type = RadImage::TYPE_ITK_FLOAT;
	}
	return;
    default:
	print_and_exit ("Error: unhandled conversion to itk_float()\n");
	return;
    }
}

void
RadImage::convert_gpuit_float ()
{
    switch (this->m_type) {
    case RadImage::TYPE_ITK_FLOAT:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    case RadImage::TYPE_GPUIT_FLOAT:
	return;
    default:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    }
}
