/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "itkImageRegionIterator.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "readmha.h"
#include "volume.h"
#include "print_and_exit.h"

/* -----------------------------------------------------------------------
   Image header conversion
   ----------------------------------------------------------------------- */
void
gpuit_direction_from_itk (
    float gpuit_direction_cosines[9],
    DirectionType* itk_direction
)
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
PlmImageHeader::set_from_gpuit (
    float gpuit_origin[3],
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
PlmImageHeader::set_from_gpuit_bspline (BSPLINE_Xform *bxf)
{
    this->set_from_gpuit (
	bxf->img_origin,
	bxf->img_spacing,
	bxf->img_dim,
	0);
}

void
PlmImageHeader::set_from_plm_image (PlmImage *pli)
{
    switch (pli->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	this->set_from_itk_image (pli->m_itk_uchar);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	this->set_from_itk_image (pli->m_itk_short);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	this->set_from_itk_image (pli->m_itk_ushort);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	this->set_from_itk_image (pli->m_itk_uint32);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	this->set_from_itk_image (pli->m_itk_float);
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
	this->set_from_itk_image (pli->m_itk_double);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
    {
	Volume* vol = (Volume*) pli->m_gpuit;
	set_from_gpuit (vol->offset, vol->pix_spacing,
			vol->dim, vol->direction_cosines);
	break;
    }
    case PLM_IMG_TYPE_ITK_FLOAT_FIELD:
    case PLM_IMG_TYPE_ITK_CHAR:
    case PLM_IMG_TYPE_ITK_LONG:
    default:
	print_and_exit ("Unhandled image type in set_from_plm_image\n");
	break;
    }
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

void 
PlmImageHeader::get_gpuit_direction_cosines (float gpuit_direction_cosines[9])
{
    gpuit_direction_from_itk (gpuit_direction_cosines, &m_direction);
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

/* Return 1 if the two headers are the same */
int
PlmImageHeader::compare (PlmImageHeader *pli1, PlmImageHeader *pli2)
{
    int d;
    for (d = 0; d < 3; d++) {
	if (pli1->m_origin[d] != pli2->m_origin[d]) return 0;
	if (pli1->m_spacing[d] != pli2->m_spacing[d]) return 0;
	if (pli1->Size(d) != pli2->Size(d)) return 0;
    }

    /* GCS FIX: check direction cosines */

    return 1;
}
