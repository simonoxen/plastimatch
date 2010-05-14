/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_h_
#define _plm_image_h_

#include "plm_config.h"
#include "itk_image.h"
#include "plm_image_type.h"
#include "plm_image_patient_position.h"
#include "print_and_exit.h"
#include "volume.h"

class Plm_image_header;
class Plm_image;

class Plm_image {

public:

    Plm_image_type m_original_type;
    Plm_image_type m_type;
    Plm_image_patient_position m_patient_pos;

    /* The actual image is one of the following. */
    UCharImageType::Pointer m_itk_uchar;
    ShortImageType::Pointer m_itk_short;
    UShortImageType::Pointer m_itk_ushort;
    Int32ImageType::Pointer m_itk_int32;
    UInt32ImageType::Pointer m_itk_uint32;
    FloatImageType::Pointer m_itk_float;
    DoubleImageType::Pointer m_itk_double;
    void* m_gpuit;

private:
    /* Please don't use copy constructors.  They suck. */
    Plm_image (Plm_image& xf) {
    }
    /* Please don't use overloaded operators.  They suck. */
    Plm_image& operator= (Plm_image& xf) {
	return *this;
    }
    plastimatch1_EXPORT
    void convert_to_itk_uchar ();
    plastimatch1_EXPORT
    void convert_to_itk_short ();
    plastimatch1_EXPORT
    void convert_to_itk_int32 (void);
    plastimatch1_EXPORT
    void convert_to_itk_uint32 ();
    plastimatch1_EXPORT
    void convert_to_itk_float ();
    plastimatch1_EXPORT
    void convert_to_gpuit_short ();
    plastimatch1_EXPORT
    void convert_to_gpuit_uint16 ();
    plastimatch1_EXPORT
    void convert_to_gpuit_uint32 ();
    plastimatch1_EXPORT
    void convert_to_gpuit_float ();

public:
    Plm_image () {
	clear ();
    }
    ~Plm_image () {
	free ();
    }

    void clear () {
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_gpuit = 0;
	m_patient_pos = PATIENT_POSITION_UNKNOWN;
    }
    void free () {
	switch (m_type) {
	case PLM_IMG_TYPE_GPUIT_FLOAT:
	case PLM_IMG_TYPE_GPUIT_FLOAT_FIELD:
	    volume_destroy ((Volume*) m_gpuit);
	    break;
	default:
	    /* GCS FIX: This doesn't actually free anything for itk. */
	    break;
	}
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_gpuit = 0;
	m_patient_pos = PATIENT_POSITION_UNKNOWN;
    }

    plastimatch1_EXPORT
    Plm_image* clone (void);

    /* Loading */
    plastimatch1_EXPORT
    void load_native (const char* fname);
    
    plastimatch1_EXPORT
    void load_native_dicom (const char* fname);
    /* Saving */
    plastimatch1_EXPORT
    void save_short_dicom (char* fname);
    plastimatch1_EXPORT
    void save_image (const char* fname);
    plastimatch1_EXPORT
    void convert_and_save (const char* fname, Plm_image_type new_type);

    /* assignment */
    plastimatch1_EXPORT
    void set_gpuit (volume *v);

    /* conversion */
    FloatImageType::Pointer& itk_float () {
	convert_to_itk_float ();
	return m_itk_float;
    }
    volume* gpuit_float () {
	convert_to_gpuit_float ();
	return (volume*) m_gpuit;
    }
    plastimatch1_EXPORT
    void convert (Plm_image_type new_type);
    plastimatch1_EXPORT
    void convert_to_original_type (void);
    plastimatch1_EXPORT
    void convert_to_itk (void);

    /* metadata */
    plastimatch1_EXPORT
    void set_metadata (char *tag, char *value);

    /* Other */
    plastimatch1_EXPORT
    static int compare_headers (Plm_image *pli1, Plm_image *pli2);
};

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT
Plm_image* plm_image_load (char* fname, Plm_image_type type);
plastimatch1_EXPORT
Plm_image* plm_image_load_native (const char* fname);
plastimatch1_EXPORT
void
plm_image_save_vol (const char* fname, Volume *vol);

#endif
