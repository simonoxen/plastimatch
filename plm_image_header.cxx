/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "itkImageRegionIterator.h"

#include "mha_io.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "print_and_exit.h"
#include "volume.h"

/* -----------------------------------------------------------------------
   prototypes
   ----------------------------------------------------------------------- */
static void
itk_direction_from_gpuit (
    DirectionType* itk_direction,
    float gpuit_direction_cosines[9]
);

static void
itk_direction_identity (DirectionType* itk_direction);

/* -----------------------------------------------------------------------
   functions
   ----------------------------------------------------------------------- */
void
Plm_image_header::set_origin (float origin[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	this->m_origin[d] = origin[d];
    }
}

void
Plm_image_header::set_spacing (float spacing[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	this->m_spacing[d] = spacing[d];
    }
}

void
Plm_image_header::set_dim (int dim[3])
{
    ImageRegionType::SizeType itk_size;
    ImageRegionType::IndexType itk_index;
    for (unsigned int d = 0; d < 3; d++) {
	itk_index[d] = 0;
	itk_size[d] = dim[d];
    }
    m_region.SetSize (itk_size);
    m_region.SetIndex (itk_index);
}

void
Plm_image_header::set_from_gpuit (
    float gpuit_origin[3],
    float gpuit_spacing[3],
    int gpuit_dim[3],
    float gpuit_direction_cosines[9])
{
    ImageRegionType::SizeType itk_size;
    ImageRegionType::IndexType itk_index;

    for (unsigned int d1 = 0; d1 < 3; d1++) {
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
Plm_image_header::set_from_gpuit_bspline (Bspline_xform *bxf)
{
    this->set_from_gpuit (
	bxf->img_origin,
	bxf->img_spacing,
	bxf->img_dim,
	0);
}

void
Plm_image_header::set_from_plm_image (Plm_image *pli)
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
    case PLM_IMG_TYPE_ITK_LONG:
	this->set_from_itk_image (pli->m_itk_int32);
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
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_GPUIT_FLOAT:
    case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
    {
	Volume* vol = (Volume*) pli->m_gpuit;
	set_from_gpuit (vol->offset, vol->pix_spacing,
			vol->dim, vol->direction_cosines);
	break;
    }
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
	this->set_from_itk_image (pli->m_itk_uchar_vec);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT_FIELD:
    case PLM_IMG_TYPE_ITK_CHAR:
    default:
	print_and_exit ("Unhandled image type (%s) in set_from_plm_image\n",
	    plm_image_type_string (pli->m_type));
	break;
    }
}

void 
Plm_image_header::get_origin (float origin[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	origin[d] = m_origin[d];
    }
}

void 
Plm_image_header::get_spacing (float spacing[3])
{
    for (unsigned int d = 0; d < 3; d++) {
	spacing[d] = m_spacing[d];
    }
}

void 
Plm_image_header::get_dim (int dim[3])
{
    ImageRegionType::SizeType itk_size = m_region.GetSize ();
    for (unsigned int d = 0; d < 3; d++) {
	dim[d] = itk_size[d];
    }
}

void 
Plm_image_header::get_direction_cosines (float direction_cosines[9])
{
    direction_cosines_from_itk (direction_cosines, &m_direction);
}

void
Plm_image_header::print (void) const
{
    ImageRegionType::SizeType itk_size;
    itk_size = m_region.GetSize ();

    printf ("Origin =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %g", m_origin[d]);
    }
    printf ("\nSize =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %lu", itk_size[d]);
    }
    printf ("\nSpacing =");
    for (unsigned int d = 0; d < 3; d++) {
	printf (" %g", m_spacing[d]);
    }
    printf ("\nDirection =");
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
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

    for (unsigned int d = 0; d < 3; d++) {
	itk_index[d] = roi_offset[d];
	itk_size[d] = roi_dim[d];
    }
    (*roi).SetSize (itk_size);
    (*roi).SetIndex (itk_index);
}

void 
Plm_image_header::get_image_center (float center[3])
{
    int d;
    for (d = 0; d < 3; d++) {
	center[d] = this->m_origin[d] 
	    + this->m_spacing[d] * (this->Size(d) - 1) / 2;
    }
}

/* Return 1 if the two headers are the same */
int
Plm_image_header::compare (Plm_image_header *pli1, Plm_image_header *pli2)
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

/* -----------------------------------------------------------------------
   global functions
   ----------------------------------------------------------------------- */
void
direction_cosines_from_itk (
    float direction_cosines[9],
    DirectionType* itk_direction
)
{
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    direction_cosines[d1*3+d2] = (*itk_direction)[d1][d2];
	}
    }
}

static void
itk_direction_from_gpuit (
    DirectionType* itk_direction,
    float gpuit_direction_cosines[9]
)
{
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    (*itk_direction)[d1][d2] = gpuit_direction_cosines[d1*3+d2];
	}
    }
}

static void
itk_direction_identity (DirectionType* itk_direction)
{
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    (*itk_direction)[d1][d2] = (float) (d1 == d2);
	}
    }
}

template<class T, class U>
void
itk_image_header_copy (T dest, U src)
{
    typedef typename U::ObjectType SrcImageType;
    typedef typename T::ObjectType DestImageType;

    const typename SrcImageType::RegionType src_rgn
	= src->GetLargestPossibleRegion();
    const typename SrcImageType::PointType& src_og = src->GetOrigin();
    //const typename SrcImageType::SizeType& src_sz = src_rgn.GetSize();
    const typename SrcImageType::SpacingType& src_sp = src->GetSpacing();
    const typename SrcImageType::DirectionType& src_dc = src->GetDirection();

    dest->SetRegions (src_rgn);
    dest->SetOrigin (src_og);
    dest->SetSpacing (src_sp);
    dest->SetDirection (src_dc);
}

/* Explicit instantiations */
template plastimatch1_EXPORT void itk_image_header_copy (UCharVecImageType::Pointer, UInt32ImageType::Pointer im_in);
