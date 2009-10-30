/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "itkImageRegionIterator.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "itk_image.h"
#include "readmha.h"
#include "volume.h"
#include "print_and_exit.h"

/* -----------------------------------------------------------------------
   Loading
   ----------------------------------------------------------------------- */
PlmImage*
plm_image_load_native (char* fname)
{
    PlmImage *pli = new PlmImage;
    if (!pli) return 0;

    pli->load_native (fname);

    return pli;
}

void
PlmImage::load_native (char* fname)
{
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    itk__GetImageType (fname, pixelType, componentType);

    switch (componentType) {
    case itk::ImageIOBase::UCHAR:
	this->m_itk_uchar = load_uchar (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_UCHAR;
	this->m_type = PLM_IMG_TYPE_ITK_UCHAR;
	break;
    case itk::ImageIOBase::SHORT:
	this->m_itk_short = load_short (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_SHORT;
	this->m_type = PLM_IMG_TYPE_ITK_SHORT;
	break;
    case itk::ImageIOBase::USHORT:
	this->m_itk_ushort = load_ushort (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_USHORT;
	this->m_type = PLM_IMG_TYPE_ITK_USHORT;
	break;
#if (CMAKE_SIZEOF_UINT == 4)
    case itk::ImageIOBase::UINT:
#endif
#if (CMAKE_SIZEOF_ULONG == 4)
    case itk::ImageIOBase::ULONG:
#endif
	this->m_itk_uint32 = load_uint32 (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_ULONG;
	this->m_type = PLM_IMG_TYPE_ITK_ULONG;
	break;
    case itk::ImageIOBase::FLOAT:
	this->m_itk_float = load_float (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_FLOAT;
	this->m_type = PLM_IMG_TYPE_ITK_FLOAT;
	break;
    default:
	printf ("Error, unsupported output type\n");
	exit (-1);
	break;
    }
}

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

/* -----------------------------------------------------------------------
   Saving
   ----------------------------------------------------------------------- */
void
PlmImage::save_short_dicom (char* fname)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save_short_dicom (this->m_itk_float, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_itk_float ();
	itk_image_save_short_dicom (this->m_itk_float, fname);
	break;
    default:
	print_and_exit ("Unhandled image type in PlmImage::save_image\n");
	break;
    }
}

void
PlmImage::save_image (char* fname)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	itk_image_save (this->m_itk_uchar, fname);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	itk_image_save (this->m_itk_short, fname);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	itk_image_save (this->m_itk_ushort, fname);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	itk_image_save (this->m_itk_uint32, fname);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save (this->m_itk_float, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_itk_float ();
	itk_image_save (this->m_itk_float, fname);
	break;
    default:
	print_and_exit ("Unhandled image type in PlmImage::save_image\n");
	break;
    }
}

/* -----------------------------------------------------------------------
   Conversion
   ----------------------------------------------------------------------- */
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

template<class T> 
static void
plm_image_convert_to_gpuit_float (PlmImage* pli, T img)
{
    typedef typename T::ObjectType ImageType;
    int i, d1;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion ();
    typename ImageType::PointType og = img->GetOrigin();
    typename ImageType::SpacingType sp = img->GetSpacing();
    typename ImageType::SizeType sz = rg.GetSize();
    typename ImageType::DirectionType dc = img->GetDirection();

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
    float* vol_img = (float*) vol->img;

    /* Copy data into gpuit */
    typedef typename itk::ImageRegionIterator< ImageType > IteratorType;
    IteratorType it (img, rg);
    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
	vol_img[i] = it.Get();
    }

#if defined (commentout)
    /* Free itk data */
    this->m_itk_float = 0;
#endif

    /* Set data type */
    pli->m_gpuit = vol;
    pli->m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
}

void
PlmImage::convert_gpuit_float ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_SHORT:
	plm_image_convert_to_gpuit_float (this, this->m_itk_short);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	plm_image_convert_to_gpuit_float (this, this->m_itk_float);
	break;
#if defined (commentout)
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
#endif
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	return;
    default:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    }
}
