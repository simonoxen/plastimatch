/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_study_metadata_h_
#define _rt_study_metadata_h_

#include "plmbase_config.h"
#include <list>

#include "itk_image_type.h"
#include "plm_image.h"
#include "plm_int.h"
#include "pstring.h"
#include "smart_pointer.h"

class Rt_study_metadata_private;
class Metadata;
class Plm_image_header;
class Slice_list;
class Volume;

class PLMBASE_API Rt_study_metadata {
public:
    SMART_POINTER_SUPPORT (Rt_study_metadata);
public:
    Rt_study_metadata_private *d_ptr;
public:
    Rt_study_metadata ();
    ~Rt_study_metadata ();
public:
    static Rt_study_metadata::Pointer load (const char* dicom_path);
public:
    const char* get_ct_series_uid () const;
    void set_ct_series_uid (const char* uid);
    const char* get_dose_instance_uid () const;
    const char* get_dose_series_uid () const;
    const char* get_frame_of_reference_uid () const;
    void set_frame_of_reference_uid (const char* uid);
    const char* get_rtss_instance_uid () const;
    const char* get_rtss_series_uid () const;
    const char* get_study_date () const;
    void set_study_date (const char* date);
    const char* get_study_time () const;
    void set_study_time (const char* time);
    const char* get_study_uid () const;
    void set_study_uid (const char* uid);

    const Plm_image_header* get_image_header () const;
    void set_image_header (const Plm_image::Pointer& pli);
    void set_image_header (const Plm_image_header& pih);
    void set_image_header (ShortImageType::Pointer img);

    const Slice_list *get_slice_list () const;
    void reset_slice_uids ();
    const char* get_slice_uid (int index) const;
    void set_slice_uid (int index, const char* slice_uid);
    bool slice_list_complete () const;
    void set_slice_list_complete ();
    int num_slices () const;

    Metadata *get_study_metadata ();
    const Metadata *get_study_metadata () const;
    void set_study_metadata (unsigned short key1, unsigned short key2,
        const std::string& val);
    Metadata *get_image_metadata ();
    const Metadata *get_image_metadata () const;
    Metadata *get_rtss_metadata ();
    const Metadata *get_rtss_metadata () const;
    Metadata *get_dose_metadata ();
    const Metadata *get_dose_metadata () const;
    void generate_new_uids ();
};

#endif
