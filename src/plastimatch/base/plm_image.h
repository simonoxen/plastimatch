/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _plm_image_h_
#define _plm_image_h_

#include "plmbase_config.h"
#include "compiler_warnings.h"
#include "itk_image.h"
#include "metadata.h"
#include "plm_image_type.h"
#include "smart_pointer.h"
#include "volume.h"

// TODO: Change type of m_meta to Metadata*

class Metadata;
class Plm_image_header;
class Plm_image;
class Plm_image_private;
class Pstring;
class Rt_study_metadata;
class Slice_index;

class PLMBASE_API Plm_image {
public:
    SMART_POINTER_SUPPORT (Plm_image);
public:
    Plm_image_private *d_ptr;
public:
    Plm_image ();
    Plm_image (const char* fname);
    Plm_image (const Pstring& fname);
    Plm_image (const std::string& fname);
    Plm_image (const char* fname, Plm_image_type type);
    Plm_image (const std::string& fname, Plm_image_type type);
    Plm_image (UCharImageType::Pointer img);
    Plm_image (ShortImageType::Pointer img);
    Plm_image (FloatImageType::Pointer img);
    Plm_image (Volume *vol);
    Plm_image (Plm_image_type type, const Plm_image_header& pih);
    ~Plm_image ();

public:
    Metadata m_meta;
    Plm_image_type m_original_type;
    Plm_image_type m_type;

    /* The actual image is one of the following. */
    CharImageType::Pointer m_itk_char;
    UCharImageType::Pointer m_itk_uchar;
    ShortImageType::Pointer m_itk_short;
    UShortImageType::Pointer m_itk_ushort;
    Int32ImageType::Pointer m_itk_int32;
    UInt32ImageType::Pointer m_itk_uint32;
    FloatImageType::Pointer m_itk_float;
    DoubleImageType::Pointer m_itk_double;
    UCharVecImageType::Pointer m_itk_uchar_vec;

private:
    /* Please don't use copy constructors.  They suck. */
    Plm_image (Plm_image& xf) {
        UNUSED_VARIABLE (xf);
    }
    /* Please don't use overloaded operators.  They suck. */
    Plm_image& operator= (Plm_image& xf) {
        UNUSED_VARIABLE (xf);
        return *this;
    }
    
    void convert_to_itk_char ();
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
    void convert_to_gpuit_int32 ();
    void convert_to_gpuit_float ();
    void convert_to_gpuit_uchar ();
    void convert_to_gpuit_uchar_vec ();

public:
    /* creation / destruction */
    void init ();
    void free ();
    bool have_image ();
    Plm_image* clone (void);
    void create (Plm_image_type type, const Plm_image_header& pih);

    /* Loading */
    bool load (const char* fname, Plm_image_type type);
    bool load_native (const char* fname);
    bool load_native (const std::string& fn);
    bool load_native_dicom (const char* fname);
    bool load_native_nki (const char* fname);

    /* Saving */
    void save_short_dicom (const char* fname, Rt_study_metadata *rsm);
    void save_image (const char* fname);
    void save_image (const Pstring& fname);
    void save_image (const std::string& fname);
    void convert_and_save (const char* fname, Plm_image_type new_type);
    void convert_and_save (const std::string& fname, Plm_image_type new_type);

    /* assignment */
    void set_volume (Volume *v, Plm_image_type type);
    void set_volume (Volume *v);
    void set_itk (UCharImageType::Pointer img);
    void set_itk (UShortImageType::Pointer img);
    void set_itk (ShortImageType::Pointer img);
    void set_itk (UInt32ImageType::Pointer img);
    void set_itk (FloatImageType::Pointer img);
    void set_itk (UCharVecImageType::Pointer img);

    /* conversion */
    FloatImageType::Pointer& itk_float () {
        convert_to_itk_float ();
        return m_itk_float;
    }
    UCharImageType::Pointer& itk_uchar () {
        convert_to_itk_uchar ();
        return m_itk_uchar;
    }
    UCharVecImageType::Pointer& itk_uchar_vec () {
        convert_to_itk_uchar_vec ();
        return m_itk_uchar_vec;
    }

    Volume* get_volume ();
    const Volume* get_volume () const;
    Volume* get_volume_uchar ();
    Volume* get_volume_short ();
    Volume::Pointer get_volume_float ();
    Volume* get_volume_uchar_vec ();

    /* To be replaced ... */
    Volume* get_volume_float_raw ();

    Volume* steal_volume ();

    void convert (Plm_image_type new_type);
    void convert_to_original_type (void);
    void convert_to_itk (void);
    void convert_to_itk_float_field (void);
    void convert_to_itk_uchar_vec (void);

    /* geometry */
    int planes ();
    size_t dim (size_t);
    float origin (size_t);
    float spacing (size_t);

    /* metadata */
    void set_metadata (char *tag, char *value);

    /* debug */
    void print ();

    /* Other */
    static int compare_headers (Plm_image *pli1, Plm_image *pli2);

protected:
    /* Utility functions */
    void free_volume ();

    /* Specific converters: implemented in plm_image_convert.cxx */
    void convert_itk_uchar_to_itk_uchar_vec ();
    void convert_itk_uint32_to_itk_uchar_vec ();
    void convert_gpuit_uint32_to_itk_uchar_vec ();
    void convert_gpuit_uchar_vec_to_itk_uchar_vec ();
    void convert_itk_uchar_vec_to_gpuit_uchar_vec ();

    /* Generic converters: implemented in plm_image_convert.cxx */
    template<class T, class U> T convert_gpuit_to_itk (Volume *vol);
    template<class T, class U> void convert_itk_to_gpuit (T img);
};

/* -----------------------------------------------------------------------
   Public functions
   ----------------------------------------------------------------------- */
PLMBASE_API Plm_image* plm_image_load (const char* fname, Plm_image_type type);
PLMBASE_API Plm_image* plm_image_load (
    const std::string& fname, Plm_image_type type);
PLMBASE_API Plm_image* plm_image_load_native (const char* fname);
PLMBASE_API Plm_image* plm_image_load_native (const std::string& fname);
PLMBASE_API void plm_image_save_vol (const char* fname, const Volume *vol);

#endif
