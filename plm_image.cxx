/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "plm_config.h"
#include "plm_image.h"
#include "itkImageRegionIterator.h"
#include "readmha.h"
#include "volume.h"
#include "print_and_exit.h"


/* -----------------------------------------------------------------------
   Image header conversion
   ----------------------------------------------------------------------- */
void
PlmImageHeader::set_from_itk (const OriginType& itk_origin,
			 const SpacingType& itk_spacing,
			 const ImageRegionType& itk_region)
{
    m_origin = itk_origin;
    m_spacing = itk_spacing;
    m_region = itk_region;
}

void
PlmImageHeader::set_from_gpuit (float gpuit_origin[3],
			 float gpuit_spacing[3],
			 int gpuit_dim[3])
{
    ImageRegionType::SizeType itk_size;
    ImageRegionType::IndexType itk_index;

    for (int d = 0; d < Dimension; d++) {
	m_origin[d] = gpuit_origin[d];
	m_spacing[d] = gpuit_spacing[d];
	itk_index[d] = 0;
	itk_size[d] = gpuit_dim[d];
    }
    m_region.SetSize (itk_size);
    m_region.SetIndex (itk_index);
}

void
PlmImageHeader::cvt_to_gpuit (float gpuit_origin[3],
			    float gpuit_spacing[3],
			    int gpuit_dim[3])
{
    ImageRegionType::SizeType itk_size;
    itk_size = m_region.GetSize ();

    for (int d = 0; d < Dimension; d++) {
	gpuit_origin[d] = m_origin[d];
	gpuit_spacing[d] = m_spacing[d];
	gpuit_dim[d] = itk_size[d];
    }
}

void
PlmImageHeader::print (void)
{
    ImageRegionType::SizeType itk_size;
    itk_size = m_region.GetSize ();

    printf ("Origin =");
    for (int d = 0; d < Dimension; d++) {
	printf (" %g", m_origin[d]);
    }
    printf ("\nSize =");
    for (int d = 0; d < Dimension; d++) {
	printf (" %g", itk_size[d]);
    }
    printf ("\nSpacing =");
    for (int d = 0; d < Dimension; d++) {
	printf (" %g", m_spacing[d]);
    }
    printf ("\nDirection =\n");
    for (int d1 = 0; d1 < Dimension; d1++) {
	for (int d2 = 0; d2 < Dimension; d2++) {
	    printf (" %g", m_direction[d1][d2]);
	}
    }
    printf ("\n");
}

void
itk_roi_from_gpuit (
    ImageRegionType* roi,
    int roi_offset[3], int roi_dim[3])
{
    ImageRegionType::SizeType itk_size;
    ImageRegionType::IndexType itk_index;

    for (int d = 0; d < Dimension; d++) {
	itk_index[d] = roi_offset[d];
	itk_size[d] = roi_dim[d];
    }
    (*roi).SetSize (itk_size);
    (*roi).SetIndex (itk_index);
}


/* -----------------------------------------------------------------------
   Image conversion
   ----------------------------------------------------------------------- */
PlmImage*
rad_image_load (char* fname, PlmImageType type)
{
    PlmImage *ri = new PlmImage;
    if (!ri) return 0;

    switch (type) {
	case PLM_IMG_TYPE_GPUIT_FLOAT:
	    ri->m_type = type;
	    ri->m_original_type = type;
	    ri->m_gpuit = read_mha (fname);
	    break;
	case PLM_IMG_TYPE_ITK_FLOAT:
	    ri->m_type = type;
	    //ri->m_itk_float = load_float (fname);
	    load_float (&ri->m_itk_float, &ri->m_original_type, fname);
	    break;
	default:
	    print_and_exit ("Unhandled image load in rad_image_load\n");
	    break;
    }
    return ri;
}

void
PlmImage::convert_itk_float ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_FLOAT:
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
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
	    this->m_type = PLM_IMG_TYPE_ITK_FLOAT;
	}
	return;
    default:
	print_and_exit ("Error: unhandled conversion to itk_float()\n");
	return;
    }
}

void
PlmImage::convert_gpuit_float ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_FLOAT:
	{
	    int i, d;
	    FloatImageType::RegionType rg = this->m_itk_float->GetLargestPossibleRegion ();
	    FloatImageType::PointType og = this->m_itk_float->GetOrigin();
	    FloatImageType::SpacingType sp = this->m_itk_float->GetSpacing();
	    FloatImageType::SizeType sz = rg.GetSize();

	    /* Copy header & allocate data for gpuit float */
	    int dim[3];
	    float offset[3];
	    float pix_spacing[3];
	    for (d = 0; d < 3; d++) {
		dim[d] = sz[d];
		offset[d] = og[d];
		pix_spacing[d] = sp[d];
	    }
	    Volume* vol = volume_create (dim, offset, pix_spacing, PT_FLOAT, 0);
	    float* img = (float*) vol->img;

	    /* Copy data into gpuit */
	    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
	    FloatIteratorType it (this->m_itk_float, rg);
	    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
		img[i] = it.Get();
	    }

	    /* Free itk data */
	    this->m_itk_float = 0;

	    /* Set data type */
	    this->m_gpuit = vol;
	    this->m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
	}
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	return;
    default:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    }
}
