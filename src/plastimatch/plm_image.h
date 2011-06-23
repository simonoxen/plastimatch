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

class Img_metadata;
class Plm_image_header;
class Plm_image;
class Referenced_dicom_dir;

class plastimatch1_EXPORT Plm_image {

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
    UCharVecImageType::Pointer m_itk_uchar_vec;
    void* m_gpuit;

private:
    /* Please don't use copy constructors.  They suck. */
    Plm_image (Plm_image& xf) {
    }
    /* Please don't use overloaded operators.  They suck. */
    Plm_image& operator= (Plm_image& xf) {
	return *this;
    }
    
    void convert_to_itk_uchar ();
    void convert_to_itk_short ();
    void convert_to_itk_ushort ();
    void convert_to_itk_int32 (void);
    void convert_to_itk_uint32 ();
    void convert_to_itk_float ();
    void convert_to_itk_double ();
    void convert_to_gpuit_short ();
    void convert_to_gpuit_uint16 ();
    void convert_to_gpuit_uint32 ();
    void convert_to_gpuit_float ();
    void convert_to_gpuit_uchar_vec ();

public:
    Plm_image () {
	this->init ();
    }
    Plm_image (const char* fname) {
	this->init ();
	this->load_native (fname);
    }
    Plm_image (const char* fname, Plm_image_type type)
    {
	this->init ();
	this->load (fname, type);
    }
    ~Plm_image () {
	this->free ();
    }

    void init () {
	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_patient_pos = PATIENT_POSITION_UNKNOWN;
	m_gpuit = 0;
    }
    void free () {
	if (m_gpuit) {
	    volume_destroy ((Volume*) m_gpuit);
	}

	m_original_type = PLM_IMG_TYPE_UNDEFINED;
	m_type = PLM_IMG_TYPE_UNDEFINED;
	m_patient_pos = PATIENT_POSITION_UNKNOWN;

	m_itk_uchar = 0;
	m_itk_short = 0;
	m_itk_ushort = 0;
	m_itk_int32 = 0;
	m_itk_uint32 = 0;
	m_itk_float = 0;
	m_itk_double = 0;
	m_gpuit = 0;
    }

    Plm_image* clone (void);

    /* Loading */
    void load (const char* fname, Plm_image_type type);
    void load_native (const char* fname);
    void load_native_dicom (const char* fname);

    /* Saving */
    void save_short_dicom (const char* fname, Referenced_dicom_dir *rdd, 
	Img_metadata *img_metadata);
    void save_image (const char* fname);
    void convert_and_save (const char* fname, Plm_image_type new_type);

    /* assignment */
    void set_gpuit (Volume *v);
    void set_itk (UCharImageType::Pointer img);
    void set_itk (UShortImageType::Pointer img);
    void set_itk (UInt32ImageType::Pointer img);
    void set_itk (FloatImageType::Pointer img);
    void set_itk (UCharVecImageType::Pointer img);

    /* conversion */
    FloatImageType::Pointer& itk_float () {
	convert_to_itk_float ();
	return m_itk_float;
    }
    /* NSh 6/22/2011 */
    UCharImageType::Pointer& itk_uchar () {
	convert_to_itk_uchar ();
	return m_itk_uchar;
    }
    Volume* gpuit_float () {
	convert_to_gpuit_float ();
	return (Volume*) m_gpuit;
    }
    Volume* gpuit_uchar_vec () {
	convert_to_gpuit_uchar_vec ();
	return (Volume*) m_gpuit;
    }
    void convert (Plm_image_type new_type);
    void convert_to_original_type (void);
    void convert_to_itk (void);
    void convert_to_itk_uchar_vec (void);

    /* metadata */
    void set_metadata (char *tag, char *value);

    /* Other */
    static int compare_headers (Plm_image *pli1, Plm_image *pli2);
};

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
plastimatch1_EXPORT
Plm_image* plm_image_load (const char* fname, Plm_image_type type);
plastimatch1_EXPORT
Plm_image* plm_image_load_native (const char* fname);
plastimatch1_EXPORT
void plm_image_save_vol (const char* fname, Volume *vol);

#endif
