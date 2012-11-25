/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"

#include "dicom_rt_study.h"
#include "dicom_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "volume.h"

class Dicom_rt_study_private {
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
//    std::vector<Dicom_slice_data>* slice_data;
public:
    Dicom_rt_study_private () {
        dicom_get_date_time (&date_string, &time_string);
        study_uid = dicom_uid (PLM_UID_PREFIX);
        for_uid = dicom_uid (PLM_UID_PREFIX);
        ct_series_uid = dicom_uid (PLM_UID_PREFIX);
        rtss_series_uid = dicom_uid (PLM_UID_PREFIX);
        rtss_instance_uid = dicom_uid (PLM_UID_PREFIX);
        dose_series_uid = dicom_uid (PLM_UID_PREFIX);
        dose_instance_uid = dicom_uid (PLM_UID_PREFIX);
//        slice_data = new std::vector<Dicom_slice_data>;
    }
    ~Dicom_rt_study_private () {
//        delete slice_data;
    }
};


Dicom_rt_study::Dicom_rt_study ()
{
    this->d_ptr = new Dicom_rt_study_private;
}

Dicom_rt_study::~Dicom_rt_study ()
{
    delete this->d_ptr;
}

const char*
Dicom_rt_study::get_ct_series_uid () const
{
    return d_ptr->ct_series_uid.c_str();
}

const char*
Dicom_rt_study::get_dose_instance_uid () const
{
    return d_ptr->dose_instance_uid.c_str();
}

const char*
Dicom_rt_study::get_dose_series_uid () const
{
    return d_ptr->dose_series_uid.c_str();
}

const char*
Dicom_rt_study::get_frame_of_reference_uid () const
{
    return d_ptr->for_uid.c_str();
}

const char*
Dicom_rt_study::get_rtss_instance_uid () const
{
    return d_ptr->rtss_instance_uid.c_str();
}

const char*
Dicom_rt_study::get_rtss_series_uid () const
{
    return d_ptr->rtss_series_uid.c_str();
}

const char*
Dicom_rt_study::get_study_date () const
{
    return d_ptr->date_string.c_str();
}

const char*
Dicom_rt_study::get_study_time () const
{
    return d_ptr->time_string.c_str();
}

const char*
Dicom_rt_study::get_study_uid () const
{
    return d_ptr->study_uid.c_str();
}
