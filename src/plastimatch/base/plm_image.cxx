/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "itkImageRegionIterator.h"
#include "itkMetaDataDictionary.h"
#include "itkMetaDataObject.h"

#include "plmbase.h"
#include "plmsys.h"

#include "dcmtk_load.h"
#include "pstring.h"

Plm_image::Plm_image () {
    this->init ();
}
Plm_image::Plm_image (const Pstring& fname)
{
    this->init ();
    this->load_native (fname.c_str());
}
Plm_image::Plm_image (const char* fname) {
    this->init ();
    this->load_native (fname);
}
Plm_image::Plm_image (const char* fname, Plm_image_type type)
{
    this->init ();
    this->load (fname, type);
}
Plm_image::~Plm_image () {
    this->free ();
}


/* -----------------------------------------------------------------------
    Creation / Destruction
   ----------------------------------------------------------------------- */
void
Plm_image::init ()
{
    m_original_type = PLM_IMG_TYPE_UNDEFINED;
    m_type = PLM_IMG_TYPE_UNDEFINED;
    m_gpuit = 0;
}

void
Plm_image::free ()
{
    if (m_gpuit) {
        delete (Volume*) m_gpuit;
    }

    m_original_type = PLM_IMG_TYPE_UNDEFINED;
    m_type = PLM_IMG_TYPE_UNDEFINED;

    m_itk_uchar = 0;
    m_itk_short = 0;
    m_itk_ushort = 0;
    m_itk_int32 = 0;
    m_itk_uint32 = 0;
    m_itk_float = 0;
    m_itk_double = 0;
    m_gpuit = 0;
}

/* -----------------------------------------------------------------------
   Cloning
   ----------------------------------------------------------------------- */
Plm_image*
Plm_image::clone (void)
{
    Plm_image *pli = new Plm_image;
    if (!pli) return 0;

    pli->m_original_type = this->m_original_type;
    pli->m_type = this->m_type;

    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	pli->m_itk_uchar = this->m_itk_uchar;
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	pli->m_itk_short = this->m_itk_short;
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	pli->m_itk_ushort = this->m_itk_ushort;
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	pli->m_itk_uint32 = this->m_itk_uint32;
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	pli->m_itk_float = this->m_itk_float;
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT16:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_GPUIT_FLOAT:
    case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
	pli->m_gpuit = (void*) volume_clone ((Volume*) this->m_gpuit);
	break;
    default:
	print_and_exit ("Unhandled image type in Plm_image::clone"
			" (type = %d)\n", this->m_type);
	break;
    }

    return pli;
}

/* -----------------------------------------------------------------------
   Loading
   ----------------------------------------------------------------------- */
Plm_image*
plm_image_load (const char* fname, Plm_image_type type)
{
    Plm_image *ri = new Plm_image;
    if (!ri) return 0;

    ri->load (fname, type);

    return ri;
}

Plm_image*
plm_image_load_native (const char* fname)
{
    Plm_image *pli = new Plm_image;
    if (!pli) return 0;

    pli->load_native (fname);

    return pli;
}

void
Plm_image::load (const char* fname, Plm_image_type type)
{
    this->free ();
    switch (type) {
    case PLM_IMG_TYPE_GPUIT_FLOAT:
        this->m_type = type;
        this->m_original_type = type;
        this->m_gpuit = read_mha (fname);
        break;
    case PLM_IMG_TYPE_ITK_FLOAT:
        this->m_type = type;
        this->m_itk_float = itk_image_load_float (fname, 
            &this->m_original_type);
        break;
    case PLM_IMG_TYPE_ITK_UCHAR:
        this->m_type = type;
        this->m_original_type = type;
	    this->m_itk_uchar = itk_image_load_uchar (fname, 0);
        break;
    default:
        print_and_exit ("Unhandled image load in plm_image_load\n");
        break;
    }
}

void
Plm_image::load_native (const char* fname)
{
    itk::ImageIOBase::IOPixelType pixel_type;
    itk::ImageIOBase::IOComponentType component_type;
    int num_dimensions, num_components;

    if (is_directory (fname)) {
	/* GCS FIX: The call to is_directory is redundant -- we already 
	   called plm_file_format_deduce() in warp_main() */
	load_native_dicom (fname);
	return;
    }

    if (!file_exists (fname)) {
	print_and_exit ("Couldn't open %s for read\n", fname);
    }

    std::string fn = fname;
    itk_image_get_props (fname, &num_dimensions, &pixel_type, 
	&component_type, &num_components);

    /* Handle ss_image as a special case */
    if (num_components > 1 && component_type == itk::ImageIOBase::UCHAR) {
	this->m_itk_uchar_vec = itk_image_load_uchar_vec (fname);
	this->m_original_type = PLM_IMG_TYPE_ITK_UCHAR_VEC;
	this->m_type = PLM_IMG_TYPE_ITK_UCHAR_VEC;
	return;
    }

    switch (component_type) {
    case itk::ImageIOBase::CHAR:
	this->m_itk_char = itk_image_load_char (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_CHAR;
	this->m_type = PLM_IMG_TYPE_ITK_CHAR;
	break;
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
    case itk::ImageIOBase::INT:
#endif
#if (CMAKE_SIZEOF_ULONG == 4)
    case itk::ImageIOBase::LONG:
#endif
	this->m_itk_int32 = itk_image_load_int32 (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_LONG;
	this->m_type = PLM_IMG_TYPE_ITK_LONG;
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
    case itk::ImageIOBase::DOUBLE:
	this->m_itk_double = itk_image_load_double (fname, 0);
	this->m_original_type = PLM_IMG_TYPE_ITK_DOUBLE;
	this->m_type = PLM_IMG_TYPE_ITK_DOUBLE;
	break;
    default:
	printf ("Error, unsupported input type in load_native(): %d\n",
	    component_type);
	exit (-1);
	break;
    }
}

void
Plm_image::load_native_dicom (const char* fname)
{
#if PLM_CONFIG_PREFER_DCMTK
    this->m_itk_short = itk_image_load_short (fname, 0);
#else
    /* GCS FIX: We don't yet have a way of getting original pixel type 
	for dicom.  Force SHORT */
    /* FIX: Patient position / direction cosines not set */
    this->m_itk_short = itk_image_load_short (fname, 0);
    this->m_original_type = PLM_IMG_TYPE_ITK_SHORT;
    this->m_type = PLM_IMG_TYPE_ITK_SHORT;
#endif
}


/* -----------------------------------------------------------------------
   Saving
   ----------------------------------------------------------------------- */
void
Plm_image::save_short_dicom (
    const char* fname, 
    Slice_index *rdd,
    Metadata *meta
)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	itk_image_save_short_dicom (this->m_itk_uchar, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	itk_image_save_short_dicom (this->m_itk_short, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	itk_image_save_short_dicom (this->m_itk_ushort, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	itk_image_save_short_dicom (this->m_itk_uint32, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save_short_dicom (this->m_itk_float, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->convert_to_itk_uchar ();
	itk_image_save_short_dicom (this->m_itk_uchar, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->convert_to_itk_short ();
	itk_image_save_short_dicom (this->m_itk_short, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->convert_to_itk_uint32 ();
	itk_image_save_short_dicom (this->m_itk_uint32, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_to_itk_float ();
	itk_image_save_short_dicom (this->m_itk_float, fname, rdd, meta);
	break;
    case PLM_IMG_TYPE_GPUIT_UINT16:
    default:
	print_and_exit ("Unhandled image type in Plm_image::save_short_dicom"
			" (type = %d)\n", this->m_type);
	break;
    }
}

void
Plm_image::save_image (const Pstring& fname)
{
    this->save_image (fname.c_str());
}

void
Plm_image::save_image (const char* fname)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_CHAR:
	itk_image_save (this->m_itk_char, fname);
	break;
    case PLM_IMG_TYPE_ITK_UCHAR:
	itk_image_save (this->m_itk_uchar, fname);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	itk_image_save (this->m_itk_short, fname);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	itk_image_save (this->m_itk_ushort, fname);
	break;
    case PLM_IMG_TYPE_ITK_LONG:
	itk_image_save (this->m_itk_int32, fname);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	itk_image_save (this->m_itk_uint32, fname);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	itk_image_save (this->m_itk_float, fname);
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
	itk_image_save (this->m_itk_double, fname);
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
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
	itk_image_save (this->m_itk_uchar_vec, fname);
	break;
    case PLM_IMG_TYPE_GPUIT_UINT16:
    default:
	print_and_exit ("Unhandled image type in Plm_image::save_image"
	    " (type = %s)\n", plm_image_type_string (this->m_type));
	break;
    }
}

/* -----------------------------------------------------------------------
   Assignment
   ----------------------------------------------------------------------- */
void 
Plm_image::set_gpuit (Volume *v)
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
    case PT_UINT16:
	m_original_type = PLM_IMG_TYPE_GPUIT_UINT16;
	m_type = PLM_IMG_TYPE_GPUIT_UINT16;
	break;
    case PT_UINT32:
	m_original_type = PLM_IMG_TYPE_GPUIT_UINT32;
	m_type = PLM_IMG_TYPE_GPUIT_UINT32;
	break;
    case PT_INT32:
	m_original_type = PLM_IMG_TYPE_GPUIT_INT32;
	m_type = PLM_IMG_TYPE_GPUIT_INT32;
	break;
    case PT_FLOAT:
	m_original_type = PLM_IMG_TYPE_GPUIT_FLOAT;
	m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
	break;
    case PT_UCHAR_VEC_INTERLEAVED:
	m_original_type = PLM_IMG_TYPE_GPUIT_UCHAR_VEC;
	m_type = PLM_IMG_TYPE_GPUIT_UCHAR_VEC;
	break;
    default:
	print_and_exit ("Undefined conversion in Plm_image::set_gpuit\n");
	break;
    }
}

void 
Plm_image::set_itk (UCharImageType::Pointer img)
{
    this->free ();
    m_original_type = PLM_IMG_TYPE_ITK_UCHAR;
    m_type = PLM_IMG_TYPE_ITK_UCHAR;
    this->m_itk_uchar = img;
}

void 
Plm_image::set_itk (UShortImageType::Pointer img)
{
    this->free ();
    m_original_type = PLM_IMG_TYPE_ITK_USHORT;
    m_type = PLM_IMG_TYPE_ITK_USHORT;
    this->m_itk_ushort = img;
}

void 
Plm_image::set_itk (UInt32ImageType::Pointer img)
{
    this->free ();
    m_original_type = PLM_IMG_TYPE_ITK_ULONG;
    m_type = PLM_IMG_TYPE_ITK_ULONG;
    this->m_itk_uint32 = img;
}

void 
Plm_image::set_itk (FloatImageType::Pointer img)
{
    this->free ();
    m_original_type = PLM_IMG_TYPE_ITK_FLOAT;
    m_type = PLM_IMG_TYPE_ITK_FLOAT;
    this->m_itk_float = img;
}

void 
Plm_image::set_itk (UCharVecImageType::Pointer img)
{
    this->free ();
    m_original_type = PLM_IMG_TYPE_ITK_UCHAR_VEC;
    m_type = PLM_IMG_TYPE_ITK_UCHAR_VEC;
    this->m_itk_uchar_vec = img;
}

/* -----------------------------------------------------------------------
   Conversion
   ----------------------------------------------------------------------- */
#define CONVERT_ITK_ITK(out_type,in_type)				\
    (this->m_itk_##out_type = cast_##out_type (this->m_itk_##in_type),	\
	this->m_itk_##in_type = 0)

void
Plm_image::convert_to_itk_char (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_CHAR:
	return;
    case PLM_IMG_TYPE_ITK_SHORT:
	CONVERT_ITK_ITK (char, short);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (char, float);
	break;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_char\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_CHAR;
}

void
Plm_image::convert_to_itk_uchar (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	return;
    case PLM_IMG_TYPE_ITK_SHORT:
	CONVERT_ITK_ITK (uchar, short);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (uchar, float);
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
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_uchar\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_UCHAR;
}

void
Plm_image::convert_to_gpuit_uchar (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
        plm_image_convert_itk_to_gpuit (this, this->m_itk_uchar, (unsigned char) 0);
        this->m_itk_uchar = 0;
        break;
    case PLM_IMG_TYPE_ITK_SHORT:
        plm_image_convert_itk_to_gpuit (this, this->m_itk_short, (unsigned char) 0);
        this->m_itk_short = 0;
        break;
    case PLM_IMG_TYPE_ITK_FLOAT:
        plm_image_convert_itk_to_gpuit (this, this->m_itk_float, (unsigned char) 0);
        this->m_itk_float = 0;
        break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
        volume_convert_to_uchar ((Volume *) this->m_gpuit);
        break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
        volume_convert_to_uchar ((Volume *) this->m_gpuit);
        break;
    default:
        print_and_exit (
            "Error: unhandled conversion from %s to itk_uchar\n",
            plm_image_type_string (this->m_type));
        return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_UCHAR;
}


void
Plm_image::convert_to_itk_short (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_SHORT:
	return;
    case PLM_IMG_TYPE_ITK_LONG:
	CONVERT_ITK_ITK (short, uint32);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (short, float);
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
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_short\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_SHORT;
}

void
Plm_image::convert_to_itk_ushort (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_SHORT:
	CONVERT_ITK_ITK (ushort, short);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	return;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (ushort, float);
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_ushort = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_ushort, (float) 0);
	break;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_ushort\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_USHORT;
}

void
Plm_image::convert_to_itk_int32 (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_ULONG:
	return;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (int32, float);
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->m_itk_int32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_int32, (unsigned char) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->m_itk_int32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_int32, (short) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->m_itk_int32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_int32, (uint32_t) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_int32 = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_int32, (float) 0);
	break;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_int32\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_ULONG;
}

void
Plm_image::convert_to_itk_uint32 (void)
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	CONVERT_ITK_ITK (uint32, uchar);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	CONVERT_ITK_ITK (uint32, short);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	return;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (uint32, float);
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
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_uint32\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_ULONG;
}

void
Plm_image::convert_to_itk_float ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	CONVERT_ITK_ITK (float, uchar);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	CONVERT_ITK_ITK (float, short);
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	CONVERT_ITK_ITK (float, ushort);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	CONVERT_ITK_ITK (float, uint32);
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
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_float\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_FLOAT;
}

void
Plm_image::convert_to_itk_double ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	CONVERT_ITK_ITK (double, uchar);
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	CONVERT_ITK_ITK (double, short);
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	CONVERT_ITK_ITK (double, uint32);
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	CONVERT_ITK_ITK (double, float);
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
	return;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
	this->m_itk_double = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_double, (unsigned char) 0);
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->m_itk_double = plm_image_convert_gpuit_to_itk (
	    this, this->m_itk_double, (float) 0);
	break;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_double\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_DOUBLE;
}

void
Plm_image::convert_to_itk_uchar_vec (void)
{
    switch (m_type) {
    case PLM_IMG_TYPE_ITK_UCHAR:
	printf ("Converting from ITK_UCHAR to ITK_UCHAR_VEC\n");
	this->m_itk_uchar_vec = plm_image_convert_itk_uchar_to_itk_uchar_vec (
	    this->m_itk_uchar);
	this->m_itk_uchar = 0;
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	printf ("Converting from ITK_ULONG to ITK_UCHAR_VEC\n");
	this->m_itk_uchar_vec = plm_image_convert_itk_uint32_to_itk_uchar_vec (
	    this->m_itk_uint32);
	this->m_itk_uint32 = 0;
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	printf ("Converting from GPUIT_UINT32 to ITK_UCHAR_VEC\n");
	this->m_itk_uchar_vec
	    = plm_image_convert_gpuit_uint32_to_itk_uchar_vec (this);
	break;
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
	break;
    case PLM_IMG_TYPE_GPUIT_UCHAR_VEC:
	printf ("Converting from GPUIT_UCHAR_VEC to ITK_UCHAR_VEC\n");
	this->m_itk_uchar_vec
	    = plm_image_convert_gpuit_uchar_vec_to_itk_uchar_vec (this);
	break;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to itk_uchar_vec\n",
	    plm_image_type_string (this->m_type));
	return;
    }
    this->m_type = PLM_IMG_TYPE_ITK_UCHAR_VEC;
}

void
Plm_image::convert_to_itk (void)
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
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
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
	print_and_exit (
	    "Error: unhandled conversion in Plm_image::convert_to_itk "
	    " with type %s.\n",
	    plm_image_type_string (this->m_type));
	break;
    }
}

void
Plm_image::convert_to_gpuit_short ()
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
	print_and_exit (
	    "Error: unhandled conversion from %s to gpuit_short\n",
	    plm_image_type_string (this->m_type));
	return;
    }
}

void
Plm_image::convert_to_gpuit_uint16 ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_GPUIT_SHORT:
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	volume_convert_to_uint16 ((Volume *) this->m_gpuit);
	return;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_ITK_UCHAR:
    case PLM_IMG_TYPE_ITK_SHORT:
    case PLM_IMG_TYPE_ITK_ULONG:
    case PLM_IMG_TYPE_ITK_FLOAT:
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to gpuit_uint16\n",
	    plm_image_type_string (this->m_type));
	return;
    }
}

void
Plm_image::convert_to_gpuit_uint32 ()
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
	print_and_exit (
	    "Error: unhandled conversion from %s to gpuit_uint32\n",
	    plm_image_type_string (this->m_type));
	return;
    }
}

void
Plm_image::convert_to_gpuit_int32 ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_GPUIT_INT32:
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	volume_convert_to_int32 ((Volume *) this->m_gpuit);
	return;
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_ITK_UCHAR:
    case PLM_IMG_TYPE_ITK_SHORT:
    case PLM_IMG_TYPE_ITK_ULONG:
    case PLM_IMG_TYPE_ITK_FLOAT:
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to gpuit_int32\n",
	    plm_image_type_string (this->m_type));
	return;
    }
}

void
Plm_image::convert_to_gpuit_float ()
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
    case PLM_IMG_TYPE_ITK_LONG:
	plm_image_convert_itk_to_gpuit_float (this, this->m_itk_int32);
	/* Free itk data */
	this->m_itk_int32 = 0;
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
    case PLM_IMG_TYPE_GPUIT_INT32:
	volume_convert_to_float ((Volume *) this->m_gpuit);
	return;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	return;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to gpuit_float\n",
	    plm_image_type_string (this->m_type));
	return;
    }
}

void
Plm_image::convert_to_gpuit_uchar_vec ()
{
    switch (this->m_type) {
    case PLM_IMG_TYPE_GPUIT_UCHAR_VEC:
	return;
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
	plm_image_convert_itk_uchar_vec_to_gpuit_uchar_vec (
	    this, this->m_itk_uchar_vec);
	/* Free itk data */
	this->m_itk_uchar_vec = 0;
	return;
    default:
	print_and_exit (
	    "Error: unhandled conversion from %s to gpuit_uchar_vec\n",
	    plm_image_type_string (this->m_type));
	return;
    }
}

void
Plm_image::convert_to_original_type (void)
{
    this->convert (this->m_original_type);
}

void
Plm_image::convert (Plm_image_type new_type)
{
    switch (new_type) {
    case PLM_IMG_TYPE_UNDEFINED:
	/* Do nothing */
	return;
    case PLM_IMG_TYPE_ITK_CHAR:
	this->convert_to_itk_char ();
	break;
    case PLM_IMG_TYPE_ITK_UCHAR:
	this->convert_to_itk_uchar ();
	break;
    case PLM_IMG_TYPE_ITK_SHORT:
	this->convert_to_itk_short ();
	break;
    case PLM_IMG_TYPE_ITK_USHORT:
	this->convert_to_itk_ushort ();
	break;
    case PLM_IMG_TYPE_ITK_LONG:
	this->convert_to_itk_int32 ();
	break;
    case PLM_IMG_TYPE_ITK_ULONG:
	this->convert_to_itk_uint32 ();
	break;
    case PLM_IMG_TYPE_ITK_FLOAT:
	this->convert_to_itk_float ();
	break;
    case PLM_IMG_TYPE_ITK_DOUBLE:
	this->convert_to_itk_double ();
	break;
    case PLM_IMG_TYPE_GPUIT_SHORT:
	this->convert_to_gpuit_short ();
	break;
    case PLM_IMG_TYPE_GPUIT_UINT16:
	this->convert_to_gpuit_uint16 ();
	break;
    case PLM_IMG_TYPE_GPUIT_UINT32:
	this->convert_to_gpuit_uint32 ();
	break;
    case PLM_IMG_TYPE_GPUIT_INT32:
	this->convert_to_gpuit_int32 ();
	break;
    case PLM_IMG_TYPE_GPUIT_FLOAT:
	this->convert_to_gpuit_float ();
	break;
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
	this->convert_to_itk_uchar_vec ();
	break;
    default:
	print_and_exit (
	    "Unhandled image type in Plm_image::convert (%s -> %s)\n", 
	    plm_image_type_string (this->m_type),
	    plm_image_type_string (new_type));
	break;
    }
    this->m_type = new_type;
}

void
Plm_image::convert_and_save (const char* fname, Plm_image_type new_type)
{
    this->convert (new_type);
    this->save_image (fname);
}

/* geometry */
int 
Plm_image::planes ()
{
    switch (m_type) {
    case PLM_IMG_TYPE_UNDEFINED:
	return 0;
    case PLM_IMG_TYPE_ITK_FLOAT_FIELD:
    case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
        return 3;
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
        return this->m_itk_uchar_vec->GetVectorLength();
    case PLM_IMG_TYPE_GPUIT_UCHAR_VEC:
        return this->vol()->vox_planes;
    default:
        return 1;
    }
}

size_t 
Plm_image::dim (size_t d)
{
    switch (m_type) {
    case PLM_IMG_TYPE_UNDEFINED:
	return 0;
    case PLM_IMG_TYPE_ITK_CHAR:
        return this->m_itk_char->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_UCHAR:
        return this->m_itk_uchar->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_SHORT:
        return this->m_itk_short->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_USHORT:
        return this->m_itk_ushort->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_LONG:
        return this->m_itk_int32->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_ULONG:
        return this->m_itk_uint32->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_FLOAT:
        return this->m_itk_float->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_ITK_DOUBLE:
        return this->m_itk_double->GetLargestPossibleRegion().GetSize()[d];
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT16:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_GPUIT_INT32:
    case PLM_IMG_TYPE_GPUIT_FLOAT:
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
    default:
	print_and_exit (
	    "Unhandled call to Plm_image::dim (type = %s)\n", 
	    plm_image_type_string (this->m_type));
	break;
    }
    return 0;
}

float 
Plm_image::origin (size_t d)
{
    switch (m_type) {
    case PLM_IMG_TYPE_UNDEFINED:
	return 0;
    case PLM_IMG_TYPE_ITK_CHAR:
        return this->m_itk_char->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_UCHAR:
        return this->m_itk_uchar->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_SHORT:
        return this->m_itk_short->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_USHORT:
        return this->m_itk_ushort->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_LONG:
        return this->m_itk_int32->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_ULONG:
        return this->m_itk_uint32->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_FLOAT:
        return this->m_itk_float->GetOrigin()[d];
    case PLM_IMG_TYPE_ITK_DOUBLE:
        return this->m_itk_double->GetOrigin()[d];
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT16:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_GPUIT_INT32:
    case PLM_IMG_TYPE_GPUIT_FLOAT:
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
    default:
	print_and_exit (
	    "Unhandled call to Plm_image::origin (type = %s)\n", 
	    plm_image_type_string (this->m_type));
	break;
    }
    return 0.f;
}

float 
Plm_image::spacing (size_t d)
{
    switch (m_type) {
    case PLM_IMG_TYPE_UNDEFINED:
	return 0;
    case PLM_IMG_TYPE_ITK_CHAR:
        return this->m_itk_char->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_UCHAR:
        return this->m_itk_uchar->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_SHORT:
        return this->m_itk_short->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_USHORT:
        return this->m_itk_ushort->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_LONG:
        return this->m_itk_int32->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_ULONG:
        return this->m_itk_uint32->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_FLOAT:
        return this->m_itk_float->GetSpacing()[d];
    case PLM_IMG_TYPE_ITK_DOUBLE:
        return this->m_itk_double->GetSpacing()[d];
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT16:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_GPUIT_INT32:
    case PLM_IMG_TYPE_GPUIT_FLOAT:
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
    default:
	print_and_exit (
	    "Unhandled call to Plm_image::spacing (type = %s)\n", 
	    plm_image_type_string (this->m_type));
	break;
    }
    return 0.f;
}

/* Printing debug information */
void
Plm_image::print ()
{
    printf ("Type = %s\n", plm_image_type_string_simple (this->m_type));
    printf ("Planes = %d\n", this->planes());
    Plm_image_header pih;
    pih.set_from_plm_image (this);
    pih.print ();
}

/* Return 1 if the two headers are the same */
int
Plm_image::compare_headers (Plm_image *pli1, Plm_image *pli2)
{
    Plm_image_header pih1, pih2;

    pih1.set_from_plm_image (pli1);
    pih2.set_from_plm_image (pli2);

    return Plm_image_header::compare (&pih1, &pih2);
}

/* Note: this works for NRRD (and dicom?), but not MHA/MHD */
/* GCS 2012-02-20: This function is not actually called anywhere, 
   but might become useful if/when we start saving metadata
   in mha/nrrd files */
void 
Plm_image::set_metadata (char *tag, char *value)
{
    itk::MetaDataDictionary *dict;

    switch (this->m_type) {
    case PLM_IMG_TYPE_ITK_ULONG:
	dict = &this->m_itk_uint32->GetMetaDataDictionary();
	itk_metadata_set (dict, tag, value);
	break;
    case PLM_IMG_TYPE_ITK_UCHAR_VEC:
	dict = &this->m_itk_uchar_vec->GetMetaDataDictionary();
	itk_metadata_set (dict, tag, value);
	break;
    default:
	print_and_exit ("Error, can't set metadata for image type %d\n",
	    this->m_type);
	break;
    }
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
    Plm_image pli;

    pli.set_gpuit (v2);
    pli.convert_to_itk ();
    pli.convert_to_itk ();
    pli.save_image (fname);
}
