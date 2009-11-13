/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_h_
#define _plm_image_h_

#include "volume.h"
#include "plm_image.h"
#include "itk_image.h"
#include "print_and_exit.h"

class PlmImageHeader;
class PlmImage;

class PlmImage {

public:

    PlmImageType m_original_type;
    PlmImageType m_type;

    /* The actual image is one of the following. */
    UCharImageType::Pointer m_itk_uchar;
    UShortImageType::Pointer m_itk_ushort;
    ShortImageType::Pointer m_itk_short;
    UInt32ImageType::Pointer m_itk_uint32;
    FloatImageType::Pointer m_itk_float;
    DoubleImageType::Pointer m_itk_double;
    void* m_gpuit;

private:
    /* Please don't use copy constructors.  They suck. */
    PlmImage (PlmImage& xf) {
    }
    /* Please don't use overloaded operators.  They suck. */
    PlmImage& operator= (PlmImage& xf) {
	return *this;
    }
    void convert_to_itk_float ();
    void convert_to_itk_uint32 ();
    void convert_to_gpuit_float ();

public:
    PlmImage () {
	clear ();
    }
    ~PlmImage () {
	free ();
    }

    void clear () {
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_gpuit = 0;
    }
    void free () {
	switch (m_type) {
	case PLM_IMG_TYPE_GPUIT_FLOAT:
	case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
	    volume_free ((Volume*) m_gpuit);
	    break;
	default:
	    /* GCS FIX: This doesn't actually free anything for itk. */
	    break;
	}
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_gpuit = 0;
    }

    /* Loading */
    void load_native (char* fname);

    /* Saving */
    void save_short_dicom (char* fname);
    void save_image (char* fname);

    /* Assignment */
    void set_gpuit_float (Volume *v) {
	free ();
	m_gpuit = (void*) v;
	m_original_type = PLM_IMG_TYPE_GPUIT_FLOAT;
	m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
    }

    /* Conversion */
    FloatImageType::Pointer& itk_float () {
	convert_to_itk_float ();
	return m_itk_float;
    }
    Volume* gpuit_float () {
	convert_to_gpuit_float ();
	return (Volume*) m_gpuit;
    }
    void convert_to_original_type (void);

    /* Other */
    static int compare_headers (PlmImage *pli1, PlmImage *pli2);
};

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
PlmImage* plm_image_load (char* fname, PlmImageType type);
PlmImage* plm_image_load_native (char* fname);

#endif
