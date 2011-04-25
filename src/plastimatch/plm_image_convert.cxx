/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImage.h"

#include "itk_image.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_int.h"

/* -----------------------------------------------------------------------
   Standard 3D image conversion
   ----------------------------------------------------------------------- */
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
	sp[d1] = vol->spacing[d1];
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
    float spacing[3];
    float direction_cosines[9];
    for (d1 = 0; d1 < 3; d1++) {
	dim[d1] = sz[d1];
	offset[d1] = og[d1];
	spacing[d1] = sp[d1];
    }
    direction_cosines_from_itk (direction_cosines, &dc);
    Volume* vol = volume_create (dim, offset, spacing, direction_cosines, 
	PT_FLOAT, 1, 0);

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

/* -----------------------------------------------------------------------
   UCharVec image conversion
   ----------------------------------------------------------------------- */
UCharVecImageType::Pointer
plm_image_convert_itk_uchar_to_itk_uchar_vec (UCharImageType::Pointer im_in)
{
    /* Create the output image */
    UCharVecImageType::Pointer im_out = UCharVecImageType::New();
    itk_image_header_copy (im_out, im_in);
    im_out->SetVectorLength (2);
    im_out->Allocate ();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< UCharImageType > UCharIteratorType;
    const UCharImageType::RegionType rgn_in 
	= im_in->GetLargestPossibleRegion();
    UCharIteratorType it_in (im_in, rgn_in);
    typedef itk::ImageRegionIterator< UCharVecImageType > UCharVecIteratorType;
    const UCharVecImageType::RegionType rgn_out
	= im_out->GetLargestPossibleRegion();
    UCharVecIteratorType it_out (im_out, rgn_out);

    itk::VariableLengthVector<unsigned char> v_out(2);
    for (it_in.GoToBegin(), it_out.GoToBegin();
	 !it_in.IsAtEnd();
	 ++it_in, ++it_out)
    {
	unsigned char v_in = it_in.Get ();
	v_out[0] = v_in;
	v_out[1] = 0;
	it_out.Set (v_out);
    }

    return im_out;
}

UCharVecImageType::Pointer
plm_image_convert_itk_uint32_to_itk_uchar_vec (UInt32ImageType::Pointer im_in)
{
    /* Create the output image */
    UCharVecImageType::Pointer im_out = UCharVecImageType::New();
    itk_image_header_copy (im_out, im_in);
    im_out->SetVectorLength (4);
    im_out->Allocate ();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< UInt32ImageType > UInt32IteratorType;
    const UInt32ImageType::RegionType rgn_in 
	= im_in->GetLargestPossibleRegion();
    UInt32IteratorType it_in (im_in, rgn_in);
    typedef itk::ImageRegionIterator< UCharVecImageType > UCharVecIteratorType;
    const UCharVecImageType::RegionType rgn_out
	= im_out->GetLargestPossibleRegion();
    UCharVecIteratorType it_out (im_out, rgn_out);

    itk::VariableLengthVector<unsigned char> v_out(4);
    for (it_in.GoToBegin(), it_out.GoToBegin();
	 !it_in.IsAtEnd();
	 ++it_in, ++it_out)
    {
	uint32_t v_in = it_in.Get ();
	v_out[0] = v_in & 0x000000FF;
	v_out[1] = (v_in & 0x0000FF00) >> 8;
	v_out[2] = (v_in & 0x00FF0000) >> 16;
	v_out[3] = (v_in & 0xFF000000) >> 24;
	it_out.Set (v_out);
    }

    return im_out;
}

UCharVecImageType::Pointer
plm_image_convert_gpuit_uint32_to_itk_uchar_vec (Plm_image* pli)
{
    int i, d;
    Volume* vol = (Volume*) pli->m_gpuit;
    uint32_t* img = (uint32_t*) vol->img;

    UCharVecImageType::Pointer im_out = UCharVecImageType::New();
    UCharVecImageType::RegionType rgn_out;
    UCharVecImageType::PointType og_out;
    UCharVecImageType::SpacingType sp_out;
    UCharVecImageType::RegionType::SizeType sz_out;
    UCharVecImageType::DirectionType dc;

    /* Copy header & allocate data for itk */
    for (d = 0; d < 3; d++) {
	sz_out[d] = vol->dim[d];
	og_out[d] = vol->offset[d];
	sp_out[d] = vol->spacing[d];
    }
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    dc[d1][d2] = vol->direction_cosines[d1*3+d2];
	}
    }
    rgn_out.SetSize (sz_out);
    im_out->SetRegions (rgn_out);
    im_out->SetOrigin (og_out);
    im_out->SetSpacing (sp_out);
    im_out->SetDirection (dc);

    /* Choose size of vectors for image */
    im_out->SetVectorLength (4);

    im_out->Allocate ();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< UCharVecImageType > IteratorType;
    IteratorType it (im_out, rgn_out);

    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it, ++i) {
	/* GCS FIX: This is probably inefficient, unless the compiler 
	   is very, very smart (which I doubt) */
	/* GCS FIX: This puts the planes in the "wrong" order, 
	   with uint32_t MSB as first component of vector */
	it.Set (itk::VariableLengthVector<unsigned char> (
		(unsigned char*) &img[i], 4));
    }

    /* Free gpuit data */
    volume_destroy (vol);
    pli->m_gpuit = 0;

    return im_out;
}

UCharVecImageType::Pointer
plm_image_convert_gpuit_uchar_vec_to_itk_uchar_vec (Plm_image* pli)
{
    int i, d;
    Volume* vol = (Volume*) pli->m_gpuit;
    unsigned char* img = (unsigned char*) vol->img;

    UCharVecImageType::Pointer im_out = UCharVecImageType::New();
    UCharVecImageType::RegionType rgn_out;
    UCharVecImageType::PointType og_out;
    UCharVecImageType::SpacingType sp_out;
    UCharVecImageType::RegionType::SizeType sz_out;
    UCharVecImageType::DirectionType dc;

    /* Copy header & allocate data for itk */
    for (d = 0; d < 3; d++) {
	sz_out[d] = vol->dim[d];
	og_out[d] = vol->offset[d];
	sp_out[d] = vol->spacing[d];
    }
    for (unsigned int d1 = 0; d1 < 3; d1++) {
	for (unsigned int d2 = 0; d2 < 3; d2++) {
	    dc[d1][d2] = vol->direction_cosines[d1*3+d2];
	}
    }
    rgn_out.SetSize (sz_out);
    im_out->SetRegions (rgn_out);
    im_out->SetOrigin (og_out);
    im_out->SetSpacing (sp_out);
    im_out->SetDirection (dc);

    /* Choose size of vectors for image, minimum of 2 planes for itk */
    int out_vec_len = vol->vox_planes;
    if (out_vec_len < 2) out_vec_len = 2;
    im_out->SetVectorLength (out_vec_len);

    im_out->Allocate ();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< UCharVecImageType > IteratorType;
    IteratorType it (im_out, rgn_out);

    itk::VariableLengthVector<unsigned char> v_out(out_vec_len);
    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it) {
	for (int j = 0; j < vol->vox_planes; ++j, ++i) {
	    v_out[j] = img[i];
	}
	it.Set (v_out);
    }

    /* Free gpuit data */
    volume_destroy (vol);
    pli->m_gpuit = 0;

    return im_out;
}

void
plm_image_convert_itk_uchar_vec_to_gpuit_uchar_vec (Plm_image* pli,
    UCharVecImageType::Pointer itk_img)
{
    /* Copy header & allocate data for gpuit image */
    int i;
    UCharVecImageType::RegionType rg = itk_img->GetLargestPossibleRegion ();
    UCharVecImageType::PointType og = itk_img->GetOrigin();
    UCharVecImageType::SpacingType sp = itk_img->GetSpacing();
    UCharVecImageType::SizeType sz = rg.GetSize();
    UCharVecImageType::DirectionType dc = itk_img->GetDirection();
    int dim[3];
    float offset[3];
    float spacing[3];
    float direction_cosines[9];
    for (int d = 0; d < 3; d++) {
	dim[d] = sz[d];
	offset[d] = og[d];
	spacing[d] = sp[d];
    }
    direction_cosines_from_itk (direction_cosines, &dc);
    int vox_planes = itk_img->GetVectorLength ();

    Volume* vol = volume_create (dim, offset, spacing, direction_cosines, 
	PT_UCHAR_VEC_INTERLEAVED, vox_planes, 0);

    unsigned char* vol_img = (unsigned char*) vol->img;

    /* Copy data into gpuit */
    typedef itk::ImageRegionIterator< UCharVecImageType > IteratorType;
    IteratorType it (itk_img, rg);
    for (it.GoToBegin(), i=0; !it.IsAtEnd(); ++it) {
	itk::VariableLengthVector<unsigned char> v = it.Get();
	for (int j = 0; j < vox_planes; ++j, ++i) {
	    vol_img[i] = v[j];
	}
    }

    /* Set data type */
    pli->m_gpuit = vol;
    pli->m_type = PLM_IMG_TYPE_GPUIT_UCHAR_VEC;
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
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, Int32ImageType::Pointer img);
template plastimatch1_EXPORT 
void
plm_image_convert_itk_to_gpuit_float (Plm_image* pli, FloatImageType::Pointer img);
