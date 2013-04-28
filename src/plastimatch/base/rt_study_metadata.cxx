/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "dicom_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "rt_study_metadata.h"
#include "slice_list.h"
#include "volume.h"

class Rt_study_metadata_private {
public:
    /* Set this if we have m_pih && ct slice uids */
    bool m_loaded;

    Plm_image_header m_pih;

    std::string date_string;
    std::string time_string;
    std::string ct_series_uid;
    std::string dose_instance_uid;
    std::string dose_series_uid;
    std::string for_uid;
    std::string rtss_instance_uid;
    std::string rtss_series_uid;
    std::string study_uid;
    Slice_list slice_list;

    Metadata study_metadata;
    Metadata image_metadata;
    Metadata rtss_metadata;
    Metadata dose_metadata;

public:
    Rt_study_metadata_private () {
        dicom_get_date_time (&date_string, &time_string);
        study_uid = dicom_uid (PLM_UID_PREFIX);
        for_uid = dicom_uid (PLM_UID_PREFIX);
        ct_series_uid = dicom_uid (PLM_UID_PREFIX);
        rtss_series_uid = dicom_uid (PLM_UID_PREFIX);
        rtss_instance_uid = dicom_uid (PLM_UID_PREFIX);
        dose_series_uid = dicom_uid (PLM_UID_PREFIX);
        dose_instance_uid = dicom_uid (PLM_UID_PREFIX);

        study_metadata.create_anonymous ();
        image_metadata.set_parent (&study_metadata);
        rtss_metadata.set_parent (&study_metadata);
        dose_metadata.set_parent (&study_metadata);
    }
public:
    void
    generate_new_uids () {
        study_uid = dicom_uid (PLM_UID_PREFIX);
        for_uid = dicom_uid (PLM_UID_PREFIX);
        ct_series_uid = dicom_uid (PLM_UID_PREFIX);
        rtss_series_uid = dicom_uid (PLM_UID_PREFIX);
        rtss_instance_uid = dicom_uid (PLM_UID_PREFIX);
        dose_series_uid = dicom_uid (PLM_UID_PREFIX);
        dose_instance_uid = dicom_uid (PLM_UID_PREFIX);
    }
};

Rt_study_metadata::Rt_study_metadata ()
{
    this->d_ptr = new Rt_study_metadata_private;
}

Rt_study_metadata::~Rt_study_metadata ()
{
    delete this->d_ptr;
}

void 
Rt_study_metadata::load (const char* dicom_path)
{
}

void 
Rt_study_metadata::save (const char* dicom_path)
{
}

const char*
Rt_study_metadata::get_ct_series_uid () const
{
    return d_ptr->ct_series_uid.c_str();
}

void
Rt_study_metadata::set_ct_series_uid (const char* uid)
{
    if (!uid) return;
    d_ptr->ct_series_uid = uid;
}

const char*
Rt_study_metadata::get_dose_instance_uid () const
{
    return d_ptr->dose_instance_uid.c_str();
}

const char*
Rt_study_metadata::get_dose_series_uid () const
{
    return d_ptr->dose_series_uid.c_str();
}

const char*
Rt_study_metadata::get_frame_of_reference_uid () const
{
    return d_ptr->for_uid.c_str();
}

void
Rt_study_metadata::set_frame_of_reference_uid (const char* uid)
{
    if (!uid) return;
    d_ptr->for_uid = uid;
}

const char*
Rt_study_metadata::get_rtss_instance_uid () const
{
    return d_ptr->rtss_instance_uid.c_str();
}

const char*
Rt_study_metadata::get_rtss_series_uid () const
{
    return d_ptr->rtss_series_uid.c_str();
}

const char*
Rt_study_metadata::get_study_date () const
{
    return d_ptr->date_string.c_str();
}

void
Rt_study_metadata::set_study_date (const char* date)
{
    if (!date) return;
    d_ptr->date_string = date;
}

const char*
Rt_study_metadata::get_study_time () const
{
    return d_ptr->time_string.c_str();
}

void
Rt_study_metadata::set_study_time (const char* time)
{
    if (!time) return;
    d_ptr->time_string = time;
}

const char*
Rt_study_metadata::get_study_uid () const
{
    return d_ptr->study_uid.c_str();
}

void
Rt_study_metadata::set_study_uid (const char* uid)
{
    if (!uid) return;
    d_ptr->study_uid = uid;
}

void
Rt_study_metadata::set_image_header (const Plm_image_header& pih)
{
    d_ptr->slice_list.set_image_header (pih);
}

const char*
Rt_study_metadata::get_slice_uid (int index) const
{
    return d_ptr->slice_list.get_slice_uid (index);
}

void 
Rt_study_metadata::set_slice_uid (int index, const char* slice_uid)
{
    if (!slice_uid) return;
    d_ptr->slice_list.set_slice_uid (index, slice_uid);
}

void 
Rt_study_metadata::set_slice_list_complete ()
{
    d_ptr->slice_list.set_slice_list_complete ();
}

const Slice_list* 
Rt_study_metadata::get_slice_list ()
{
    return &d_ptr->slice_list;
}

int 
Rt_study_metadata::num_slices ()
{
    return d_ptr->slice_list.num_slices ();
}

Metadata*
Rt_study_metadata::get_study_metadata ()
{
    return &d_ptr->study_metadata;
}

const Metadata*
Rt_study_metadata::get_study_metadata () const
{
    return &d_ptr->study_metadata;
}

Metadata*
Rt_study_metadata::get_image_metadata ()
{
    return &d_ptr->image_metadata;
}

const Metadata*
Rt_study_metadata::get_image_metadata () const
{
    return &d_ptr->image_metadata;
}

Metadata*
Rt_study_metadata::get_rtss_metadata ()
{
    return &d_ptr->rtss_metadata;
}

const Metadata*
Rt_study_metadata::get_rtss_metadata () const
{
    return &d_ptr->rtss_metadata;
}

Metadata*
Rt_study_metadata::get_dose_metadata ()
{
    return &d_ptr->dose_metadata;
}

const Metadata*
Rt_study_metadata::get_dose_metadata () const
{
    return &d_ptr->dose_metadata;
}

void
Rt_study_metadata::generate_new_uids () 
{
    d_ptr->generate_new_uids ();
}
