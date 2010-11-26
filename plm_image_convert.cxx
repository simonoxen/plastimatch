/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImage.h"

#include "itk_image.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"

template<class T, class U> 
T
plm_image_convert_gpuit_to_itk (Plm_image* pli, T itk_img, U)
{
    typedef typename T::ObjectType ImageType;
    int i, d1, d2;
    Volume* vol = (Volume*) pli->m_gpuit;
    U* img = (U*) vol->img;
    typename ImageType::SizeType sz;
    typename ImageType::IndexType st;
    typename ImageType::RegionType rg;
    typename ImageType::PointType og;
    typename ImageType::SpacingType sp;
    typename ImageType::DirectionType dc;

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

    itk_img = ImageType::New();
    itk_img->SetRegions (rg);
    itk_img->SetOrigin (og);
    itk_img->SetSpacing (sp);
    itk_img->SetDirection (dc);

    itk_img->Allocate();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    IteratorType it (itk_img, rg);
    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
	/* Type conversion: U -> itk happens here */
	it.Set (img[i]);
    }

    /* Free gpuit data */
    volume_destroy (vol);
    pli->m_gpuit = 0;

    return itk_img;
}

template<class U>
UCharImage4DType::Pointer
plm_image_convert_gpuit_to_itk_uchar_4d (Plm_image* pli, U)
{
    typedef UCharImage4DType ImageType;
    int i, d;
#if defined (commentout)
    int d1, d2;
#endif
    Volume* vol = (Volume*) pli->m_gpuit;
    U* img = (U*) vol->img;

    UCharImage4DType::Pointer im_out = UCharImage4DType::New();
    UCharImage4DType::RegionType rgn_out;
    UCharImage4DType::PointType og_out;
    UCharImage4DType::SpacingType sp_out;
    //UCharImage4DType::RegionType::IndexType idx_out;
    UCharImage4DType::RegionType::SizeType sz_out;

    /* GCS FIX: What to do about directiontype??? The 4D image struct 
       messes it up. */
    UCharImage4DType::DirectionType dc;

    /* Copy header & allocate data for itk */
    sz_out[0] = 1;
    og_out[0] = 0;
    sp_out[0] = 1;
    for (d = 0; d < 3; d++) {
	sz_out[d+1] = vol->dim[d];
	og_out[d+1] = vol->offset[d];
	sp_out[d+1] = vol->pix_spacing[d];
    }
#if defined (commentout)
    for (d1 = 0; d1 < 3; d1++) {
	for (d2 = 0; d2 < 3; d2++) {
	    dc[d1][d2] = vol->direction_cosines[d1*3+d2];
	}
    }
#endif
    rgn_out.SetSize (sz_out);
    im_out->SetRegions (rgn_out);
    im_out->SetOrigin (og_out);
    im_out->SetSpacing (sp_out);
#if defined (commentout)
    im_out->SetDirection (dc);
#endif
    im_out->Allocate ();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    IteratorType it (im_out, rgn_out);
    /* GCS FIX: This needs to e.g. convert 32-bit int into 4 uchar planes */
    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
	/* Type conversion: U -> itk happens here */
	it.Set (img[i]);
    }

    /* Free gpuit data */
    volume_destroy (vol);
    pli->m_gpuit = 0;

    return im_out;
}

template<class T> 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, T img)
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
    direction_cosines_from_itk (direction_cosines, &dc);
    Volume* vol = volume_create (dim, offset, pix_spacing, PT_FLOAT, 
				 direction_cosines, 0);
    float* vol_img = (float*) vol->img;

    /* Copy data into gpuit */
    typedef typename itk::ImageRegionIterator< ImageType > IteratorType;
    IteratorType it (img, rg);
    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
	vol_img[i] = it.Get();
    }

    /* Set data type */
    pli->m_gpuit = vol;
    pli->m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
}

template <class T>
UCharImage4DType::Pointer
plm_image_convert_itk_to_itk_uchar_4d (T im_in)
{
    int d;
    UCharImage4DType::Pointer im_out = UCharImage4DType::New();

    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;

    typename ImageType::RegionType rgn_in = im_in->GetLargestPossibleRegion();
    const typename ImageType::PointType& og_in = im_in->GetOrigin();
    const typename ImageType::SpacingType& sp_in = im_in->GetSpacing();

    UCharImage4DType::RegionType rgn_out;
    UCharImage4DType::PointType og_out;
    UCharImage4DType::SpacingType sp_out;
    //UCharImage4DType::RegionType::IndexType idx_out;
    UCharImage4DType::RegionType::SizeType sz_out;

    sz_out[0] = 1;
    og_out[0] = 0;
    sp_out[0] = 1;
    for (d = 0; d < 3; d++) {
	sz_out[d+1] = rgn_in.GetSize()[d];
	og_out[d+1] = og_in[d];
	sp_out[d+1] = sp_in[d];
    }
    rgn_out.SetSize (sz_out);
    im_out->SetRegions (rgn_out);
    im_out->SetOrigin (og_out);
    im_out->SetSpacing (sp_out);
    im_out->Allocate ();

    return im_out;
}

/* Explicit instantiations */
template plastimatch1_EXPORT 
UCharImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UCharImageType::Pointer itk_img, unsigned char);
template plastimatch1_EXPORT 
UCharImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UCharImageType::Pointer itk_img, float);
template plastimatch1_EXPORT 
ShortImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, ShortImageType::Pointer itk_img, short);
template plastimatch1_EXPORT 
ShortImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, ShortImageType::Pointer itk_img, float);
template plastimatch1_EXPORT 
UShortImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UShortImageType::Pointer itk_img, unsigned short);
template plastimatch1_EXPORT 
UShortImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UShortImageType::Pointer itk_img, float);
template plastimatch1_EXPORT 
Int32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, Int32ImageType::Pointer itk_img, unsigned char);
template plastimatch1_EXPORT 
Int32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, Int32ImageType::Pointer itk_img, short);
template plastimatch1_EXPORT 
Int32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, Int32ImageType::Pointer itk_img, uint32_t);
template plastimatch1_EXPORT 
Int32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, Int32ImageType::Pointer itk_img, float);
template plastimatch1_EXPORT 
UInt32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UInt32ImageType::Pointer itk_img, unsigned char);
template plastimatch1_EXPORT 
UInt32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UInt32ImageType::Pointer itk_img, short);
template plastimatch1_EXPORT 
UInt32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UInt32ImageType::Pointer itk_img, uint32_t);
template plastimatch1_EXPORT 
UInt32ImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, UInt32ImageType::Pointer itk_img, float);
template plastimatch1_EXPORT 
FloatImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, FloatImageType::Pointer itk_img, unsigned char);
template plastimatch1_EXPORT 
FloatImageType::Pointer
plm_image_convert_gpuit_to_itk (Plm_image* pli, FloatImageType::Pointer itk_img, float);

template plastimatch1_EXPORT 
UCharImage4DType::Pointer
plm_image_convert_gpuit_to_itk_uchar_4d (Plm_image* pli, uint32_t);

template plastimatch1_EXPORT 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, UCharImageType::Pointer img);
template plastimatch1_EXPORT 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, ShortImageType::Pointer img);
template plastimatch1_EXPORT 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, UInt32ImageType::Pointer img);
template plastimatch1_EXPORT 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, FloatImageType::Pointer img);

template plastimatch1_EXPORT UCharImage4DType::Pointer plm_image_convert_itk_to_itk_uchar_4d (UInt32ImageType::Pointer);
