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
gpuit_direction_from_itk (
    float gpuit_direction_cosines[9],
    DirectionType* itk_direction)	    
{
    for (unsigned int d1 = 0; d1 < Dimension; d1++) {
	for (unsigned int d2 = 0; d2 < Dimension; d2++) {
	    gpuit_direction_cosines[d1*3+d2] = (*itk_direction)[d1][d2];
	}
    }
}

void
itk_direction_from_gpuit (
    DirectionType* itk_direction,
    float gpuit_direction_cosines[9])
{
    for (unsigned int d1 = 0; d1 < Dimension; d1++) {
	for (unsigned int d2 = 0; d2 < Dimension; d2++) {
	    (*itk_direction)[d1][d2] = gpuit_direction_cosines[d1*3+d2];
	}
    }
}

void
itk_direction_identity (
    DirectionType* itk_direction)
{
    for (unsigned int d1 = 0; d1 < Dimension; d1++) {
	for (unsigned int d2 = 0; d2 < Dimension; d2++) {
	    (*itk_direction)[d1][d2] = (float) (d1 == d2);
	}
    }
}

void
PlmImageHeader::set_from_gpuit (float gpuit_origin[3],
			 float gpuit_spacing[3],
			 int gpuit_dim[3],
			 float gpuit_direction_cosines[9])
{
    ImageRegionType::SizeType itk_size;
    ImageRegionType::IndexType itk_index;

    for (unsigned int d1 = 0; d1 < Dimension; d1++) {
	m_origin[d1] = gpuit_origin[d1];
	m_spacing[d1] = gpuit_spacing[d1];
	itk_index[d1] = 0;
	itk_size[d1] = gpuit_dim[d1];
    }
    if (gpuit_direction_cosines) {
	itk_direction_from_gpuit (&m_direction, gpuit_direction_cosines);
    } else {
	itk_direction_identity (&m_direction);
    }
    m_region.SetSize (itk_size);
    m_region.SetIndex (itk_index);
}

void 
PlmImageHeader::get_gpuit_origin (float gpuit_origin[3])
{
    for (unsigned int d = 0; d < Dimension; d++) {
	gpuit_origin[d] = m_origin[d];
    }
}

void 
PlmImageHeader::get_gpuit_spacing (float gpuit_spacing[3])
{
    for (unsigned int d = 0; d < Dimension; d++) {
	gpuit_spacing[d] = m_spacing[d];
    }
}

void 
PlmImageHeader::get_gpuit_dim (int gpuit_dim[3])
{
    ImageRegionType::SizeType itk_size = m_region.GetSize ();
    for (unsigned int d = 0; d < Dimension; d++) {
	gpuit_dim[d] = itk_size[d];
    }
}

#if defined (commentout)
void
PlmImageHeader::cvt_to_gpuit (float gpuit_origin[3],
			    float gpuit_spacing[3],
			    int gpuit_dim[3],
			    float gpuit_direction_cosines[9])
{
    ImageRegionType::SizeType itk_size;
    itk_size = m_region.GetSize ();

    for (int d1 = 0; d1 < Dimension; d1++) {
	gpuit_origin[d1] = m_origin[d1];
	gpuit_spacing[d1] = m_spacing[d1];
	gpuit_dim[d1] = itk_size[d1];
    }
    gpuit_direction_from_itk (gpuit_direction_cosines, &m_direction);
}
#endif

void
PlmImageHeader::print (void)
{
    ImageRegionType::SizeType itk_size;
    itk_size = m_region.GetSize ();

    printf ("Origin =");
    for (unsigned int d = 0; d < Dimension; d++) {
	printf (" %g", m_origin[d]);
    }
    printf ("\nSize =");
    for (unsigned int d = 0; d < Dimension; d++) {
	printf (" %lu", itk_size[d]);
    }
    printf ("\nSpacing =");
    for (unsigned int d = 0; d < Dimension; d++) {
	printf (" %g", m_spacing[d]);
    }
    printf ("\nDirection =\n");
    for (unsigned int d1 = 0; d1 < Dimension; d1++) {
	for (unsigned int d2 = 0; d2 < Dimension; d2++) {
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

    for (unsigned int d = 0; d < Dimension; d++) {
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
plm_image_load (char* fname, PlmImageType type)
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
	    //load_float (&ri->m_itk_float, &ri->m_original_type, fname);
	    ri->m_itk_float = load_float (fname, &ri->m_original_type);
	    break;
	default:
	    print_and_exit ("Unhandled image load in plm_image_load\n");
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
	    int i, d1, d2;
	    Volume* vol = (Volume*) m_gpuit;
	    float* img = (float*) vol->img;
	    FloatImageType::SizeType sz;
	    FloatImageType::IndexType st;
	    FloatImageType::RegionType rg;
	    FloatImageType::PointType og;
	    FloatImageType::SpacingType sp;
	    FloatImageType::DirectionType dc;

	    /* Copy header & allocate data for itk */
	    for (d1 = 0; d1 < 3; d1++) {
		st[d1] = 0;
		sz[d1] = vol->dim[d1];
		sp[d1] = vol->pix_spacing[d1];
		og[d1] = vol->offset[d1];
		for (d2 = 0; d2 < 3; d2++) {
		    dc[d1][d2] = vol->direction_cosines[d1*3+d2];
		}
	    }
	    rg.SetSize (sz);
	    rg.SetIndex (st);

	    this->m_itk_float = FloatImageType::New();
	    this->m_itk_float->SetRegions (rg);
	    this->m_itk_float->SetOrigin (og);
	    this->m_itk_float->SetSpacing (sp);
	    this->m_itk_float->SetDirection (dc);
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
	    int i, d1;
	    FloatImageType::RegionType rg = this->m_itk_float->GetLargestPossibleRegion ();
	    FloatImageType::PointType og = this->m_itk_float->GetOrigin();
	    FloatImageType::SpacingType sp = this->m_itk_float->GetSpacing();
	    FloatImageType::SizeType sz = rg.GetSize();
	    FloatImageType::DirectionType dc = this->m_itk_float->GetDirection();

	    /* Copy header & allocate data for gpuit float */
	    int dim[3];
	    float offset[3];
	    float pix_spacing[3];
	    float direction_cosines[9];
	    for (d1 = 0; d1 < 3; d1++) {
		dim[d1] = sz[d1];
		offset[d1] = og[d1];
		pix_spacing[d1] = sp[d1];
	    }
	    gpuit_direction_from_itk (direction_cosines, &dc);
	    Volume* vol = volume_create (dim, offset, pix_spacing, PT_FLOAT, 
		direction_cosines, 0);
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
