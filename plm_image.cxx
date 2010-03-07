/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "itkImageRegionIterator.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"

#include "file_util.h"
#include "itk_image.h"
#include "itk_image_cast.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "print_and_exit.h"
#include "readmha.h"
#include "volume.h"

/* -----------------------------------------------------------------------
   Loading
   ----------------------------------------------------------------------- */
PlmImage*
plm_image_load_native (const char* fname)
{
    PlmImage *pli = new PlmImage;
    if (!pli) return 0;

    pli->load_native (fname);

    return pli;
}

void
PlmImage::load_native_dicom (const char* fname)
{
    /* GCS FIX: We don't yet have a way of getting original pixel type 
	for dicom.  Force SHORT */
    this->m_itk_short = itk_image_load_short (fname, 0);
    this->m_original_type = PLM_IMG_TYPE_ITK_SHORT;
    this->m_type = PLM_IMG_TYPE_ITK_SHORT;
}

void
PlmImage::load_native (const char* fname)
{
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;

    if (is_directory (fname)) {
	/* GCS FIX: The call to is_directory is redundant -- we already 
	    called deduce_file_type in warp_main() */
	load_native_dicom (fname);
	return;
    }

    itk__GetImageType (fname, pixelType, componentType);

    switch (componentType) {
    case itk::ImageIOBase::UCHAR:
	this->m_itk_uchar = itk_image_load_uchar (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_UCHAR;
	this->m_type = PLM_IMG_TYPE_ITK_UCHAR;
	break;
    case itk::ImageIOBase::SHORT:
	this->m_itk_short = itk_image_load_short (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_SHORT;
	this->m_type = PLM_IMG_TYPE_ITK_SHORT;
	break;
    case itk::ImageIOBase::USHORT:
	this->m_itk_ushort = itk_image_load_ushort (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_USHORT;
	this->m_type = PLM_IMG_TYPE_ITK_USHORT;
	break;
#if (CMAKE_SIZEOF_UINT == 4)
    case itk::ImageIOBase::UINT:
#endif
#if (CMAKE_SIZEOF_ULONG == 4)
    case itk::ImageIOBase::ULONG:
#endif
	this->m_itk_uint32 = itk_image_load_uint32 (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_ULONG;
	this->m_type = PLM_IMG_TYPE_ITK_ULONG;
	break;
    case itk::ImageIOBase::FLOAT:
	this->m_itk_float = itk_image_load_float (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_FLOAT;
	this->m_type = PLM_IMG_TYPE_ITK_FLOAT;
	break;
    default:
	printf ("Error, unsupported input type in load_native(): %d\n",
	    componentType);
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
	    ri->m_itk_float = itk_image_load_float (fname, 
		&ri->m_original_type);
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
    case PLM_IMG_TYPE_ITK_UCHAR:
	itk_image_save_short_dicom (this->m_itk_uchar, fname);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	itk_image_save_short_dicom (this->m_itk_short, fname);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	itk_image_save_short_dicom (this->m_itk_ushort, fname);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	itk_image_save_short_dicom (this->m_itk_uint32, fname);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save_short_dicom (this->m_itk_float, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_to_itk_float ();
	itk_image_save_short_dicom (this->m_itk_float, fname);
	break;
    default:
	print_and_exit ("Unhandled image type in PlmImage::save_short_dicom"
			" (type = %d)\n", this->m_type);
	break;
    }
}

void
PlmImage::save_image (const char* fname)
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
	this->set_metadata ("Hello", "World");
	itk_image_save (this->m_itk_uint32, fname);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save (this->m_itk_float, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->convert_to_itk_uchar ();
	itk_image_save (this->m_itk_uchar, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->convert_to_itk_short ();
	itk_image_save (this->m_itk_short, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->convert_to_itk_uint32 ();
	itk_image_save (this->m_itk_uint32, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_to_itk_float ();
	itk_image_save (this->m_itk_float, fname);
	break;
    default:
	print_and_exit ("Unhandled image type in PlmImage::save_image"
			" (type = %d)\n", this->m_type);
	break;
    }
}

/* -----------------------------------------------------------------------
   Assignment
   ----------------------------------------------------------------------- */
void 
PlmImage::set_gpuit (volume *v)
{
    this->free ();
    m_gpuit = (void*) v;
    switch (v->pix_type) {
    case PT_UCHAR:
	m_original_type = PLM_IMG_TYPE_GPUIT_UCHAR;
	m_type = PLM_IMG_TYPE_GPUIT_UCHAR;
	break;
    case PT_SHORT:
	m_original_type = PLM_IMG_TYPE_GPUIT_SHORT;
	m_type = PLM_IMG_TYPE_GPUIT_SHORT;
	break;
    case PT_UINT32:
	m_original_type = PLM_IMG_TYPE_GPUIT_UINT32;
	m_type = PLM_IMG_TYPE_GPUIT_UINT32;
	break;
    case PT_FLOAT:
	m_original_type = PLM_IMG_TYPE_GPUIT_FLOAT;
	m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
	break;
    default:
	print_and_exit ("Undefined conversion in Plm_image::set_gpuit\n");
	break;
    }
}

/* -----------------------------------------------------------------------
   Conversion
   ----------------------------------------------------------------------- */
template<class T, class U> 
static T
plm_image_convert_gpuit_to_itk (PlmImage* pli, T itk_img, U)
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

template<class T> 
static void
plm_image_convert_itk_to_gpuit_float (PlmImage* pli, T img)
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

    /* Set data type */
    pli->m_gpuit = vol;
    pli->m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
}

void
PlmImage::convert_to_itk_uchar (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	return;
    case PLM_IMG_TYPE_ITK_FLOAT:
	this->m_itk_uchar = cast_uchar (this->m_itk_float);
	this->m_itk_float = 0;
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->m_itk_uchar = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_uchar, (unsigned char) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_uchar = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_uchar, float (0));
	break;
    default:
	print_and_exit ("Error: unhandled conversion to itk_uchar\n");
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_UCHAR;
}

void
PlmImage::convert_to_itk_short (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_SHORT:
	return;
    case PLM_IMG_TYPE_ITK_FLOAT:
	this->m_itk_short = cast_short (this->m_itk_float);
	this->m_itk_float = 0;
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->m_itk_short = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_short, (short) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_short = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_short, (float) 0);
	break;
    default:
	print_and_exit ("Error: unhandled conversion to itk_short\n");
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_SHORT;
}

void
PlmImage::convert_to_itk_uint32 (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_ULONG:
	return;
    case PLM_IMG_TYPE_ITK_FLOAT:
	this->m_itk_uint32 = cast_uint32 (this->m_itk_float);
	this->m_itk_float = 0;
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->m_itk_uint32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_uint32, (unsigned char) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->m_itk_uint32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_uint32, (short) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->m_itk_uint32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_uint32, (uint32_t) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_uint32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_uint32, (float) 0);
	break;
    default:
	print_and_exit ("Error: unhandled conversion to itk_uint32\n");
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_ULONG;
}

void
PlmImage::convert_to_itk_float ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	this->m_itk_float = cast_float (this->m_itk_uchar);
	this->m_itk_uchar = 0;
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	this->m_itk_float = cast_float (this->m_itk_short);
	this->m_itk_short = 0;
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	this->m_itk_float = cast_float (this->m_itk_uint32);
	this->m_itk_uint32 = 0;
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	return;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->m_itk_float = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_float, (unsigned char) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_float = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_float, (float) 0);
	break;
    default:
	print_and_exit ("Error: unhandled conversion to itk_float\n");
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_FLOAT;
}

void
PlmImage::convert_to_itk (void)
{
    switch (m_type) {

    case PLM_IMG_TYPE_ITK_CHAR:
    case PLM_IMG_TYPE_ITK_UCHAR:
    case PLM_IMG_TYPE_ITK_SHORT:
    case PLM_IMG_TYPE_ITK_USHORT:
    case PLM_IMG_TYPE_ITK_LONG:
    case PLM_IMG_TYPE_ITK_ULONG:
    case PLM_IMG_TYPE_ITK_FLOAT:
    case PLM_IMG_TYPE_ITK_DOUBLE:
    case PLM_IMG_TYPE_ITK_FLOAT_FIELD:
	/* Do nothing */
	break;

    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->convert_to_itk_uchar ();
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->convert_to_itk_short ();
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->convert_to_itk_uint32 ();
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_to_itk_float ();
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
    default:
	print_and_exit ("Undefined conversion in Plm_image::convert_to_itk\n");
	break;
    }
}

void
PlmImage::convert_to_gpuit_short ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_GPUIT_SHORT:
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	volume_convert_to_short ((Volume *) this->m_gpuit);
	return;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_ITK_UCHAR:
    case PLM_IMG_TYPE_ITK_SHORT:
    case PLM_IMG_TYPE_ITK_ULONG:
    case PLM_IMG_TYPE_ITK_FLOAT:
    default:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    }
}

void
PlmImage::convert_to_gpuit_uint32 ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_GPUIT_UINT32:
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	volume_convert_to_uint32 ((Volume *) this->m_gpuit);
	return;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_ITK_UCHAR:
    case PLM_IMG_TYPE_ITK_SHORT:
    case PLM_IMG_TYPE_ITK_ULONG:
    case PLM_IMG_TYPE_ITK_FLOAT:
    default:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    }
}

void
PlmImage::convert_to_gpuit_float ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	plm_image_convert_itk_to_gpuit_float (this, this->m_itk_uchar);
	/* Free itk data */
	this->m_itk_short = 0;
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	plm_image_convert_itk_to_gpuit_float (this, this->m_itk_short);
	/* Free itk data */
	this->m_itk_short = 0;
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	plm_image_convert_itk_to_gpuit_float (this, this->m_itk_uint32);
	/* Free itk data */
	this->m_itk_uint32 = 0;
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	plm_image_convert_itk_to_gpuit_float (this, this->m_itk_float);
	/* Free itk data */
	this->m_itk_float = 0;
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT32:
	volume_convert_to_float ((Volume *) this->m_gpuit);
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	return;
    default:
	print_and_exit ("Error: unhandled conversion to gpuit_float()\n");
	return;
    }
}

void
PlmImage::convert_to_original_type (void)
{
    this->convert (this->m_original_type);
}

void
PlmImage::convert (PlmImageType new_type)
{
    switch (new_type) {
    case PLM_IMG_TYPE_UNDEFINED:
	/* Do nothing */
	return;
    case PLM_IMG_TYPE_ITK_UCHAR:
	this->convert_to_itk_uchar ();
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	this->convert_to_itk_short ();
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	this->convert_to_itk_uint32 ();
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	this->convert_to_itk_float ();
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->convert_to_gpuit_short ();
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->convert_to_gpuit_uint32 ();
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_to_gpuit_float ();
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
    default:
	print_and_exit (
	    "Unhandled image type in PlmImage::convert (%d -> %d)\n", 
	    this->m_type, new_type);
	break;
    }
    this->m_type = new_type;
}

void
PlmImage::convert_and_save (const char* fname, PlmImageType new_type)
{
    this->convert (new_type);
    this->save_image (fname);
}

/* Return 1 if the two headers are the same */
int
PlmImage::compare_headers (PlmImage *pli1, PlmImage *pli2)
{
    PlmImageHeader pih1, pih2;

    pih1.set_from_plm_image (pli1);
    pih2.set_from_plm_image (pli2);

    return PlmImageHeader::compare (&pih1, &pih2);
}

void 
PlmImage::set_metadata (char *tag, char *value)
{
    /* GCS FIX: This works for NRRD (and dicom?), but not MHA/MHD */
#if defined (commentout)
    typedef itk::MetaDataObject< std::string > MetaDataStringType;

    itk::MetaDataDictionary *dict;

    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_ULONG:
	{
	    printf ("SETTING METADATA ????\n");
	    dict = &this->m_itk_uint32->GetMetaDataDictionary();

	    itk::EncapsulateMetaData<std::string> (
		*dict, std::string (tag), std::string (value));

	    itk::MetaDataDictionary::ConstIterator itr = dict->Begin();
	    itk::MetaDataDictionary::ConstIterator end = dict->End();

	    while ( itr != end ) {
		itk::MetaDataObjectBase::Pointer entry = itr->second;
		MetaDataStringType::Pointer entryvalue =
		    dynamic_cast<MetaDataStringType *>( entry.GetPointer());
		if (entryvalue) {
		    std::string tagkey = itr->first;
		    std::string tagvalue = entryvalue->GetMetaDataObjectValue();
		    std::cout << tagkey << " = " << tagvalue << std::endl;
		}
		++itr;
	    }
	}
	break;
    default:
	print_and_exit ("Error, can't set metadata for image type %d\n",
	    this->m_type);
	break;
    }
#endif
}

/* GCS FIX:  This is inefficient.  Because the pli owns the vol, 
   it will free it when it converts to itk.  Therefore we make an 
   extra copy just for this deletion.  Maybe we could switch to 
   reference counting?  See e.g. 
   http://blog.placidhacker.com/2008/11/reference-counting-in-c.html
   for an example of ref counting in C.  */
void
plm_image_save_vol (const char* fname, Volume *vol)
{
    Volume *v2 = volume_clone (vol);
    PlmImage pli;

    pli.set_gpuit (v2);
    pli.convert_to_itk ();
    pli.convert_to_itk ();
    pli.save_image (fname);
}
