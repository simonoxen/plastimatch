/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_image.h"
#include "dcmtk_rt_study.h"
#include "dcmtk_rtss.h"
#include "dcmtk_save.h"
#include "dcmtk_series.h"
#include "dcmtk_slice_data.h"
#include "dcmtk_uid.h"
#include "plm_image.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "volume.h"

class Dcmtk_rt_study_private {
public:
    OFString date_string;
    OFString time_string;
    char ct_series_uid[100];
    char dose_instance_uid[100];
    char dose_series_uid[100];
    char for_uid[100];
    char rtss_instance_uid[100];
    char rtss_series_uid[100];
    char study_uid[100];
    std::vector<Dcmtk_slice_data>* slice_data;

    
    Dcmtk_series *ds_rtdose;
    Dcmtk_series *ds_rtss;

    Rtss_structure_set *cxt;
    Metadata *cxt_metadata;
    Plm_image::Pointer img;
    Plm_image::Pointer dose;

public:
    Dcmtk_rt_study_private () {
        DcmDate::getCurrentDate (date_string);
        DcmTime::getCurrentTime (time_string);
        dcmtk_uid (study_uid, PLM_UID_PREFIX);
        dcmtk_uid (for_uid, PLM_UID_PREFIX);
        dcmtk_uid (ct_series_uid, PLM_UID_PREFIX);
        dcmtk_uid (rtss_series_uid, PLM_UID_PREFIX);
        dcmtk_uid (rtss_instance_uid, PLM_UID_PREFIX);
        dcmtk_uid (dose_series_uid, PLM_UID_PREFIX);
        dcmtk_uid (dose_instance_uid, PLM_UID_PREFIX);
        slice_data = new std::vector<Dcmtk_slice_data>;
    }
    ~Dcmtk_rt_study_private () {
        delete slice_data;
    }
};


Dcmtk_rt_study::Dcmtk_rt_study ()
{
    this->d_ptr = new Dcmtk_rt_study_private;
}

Dcmtk_rt_study::~Dcmtk_rt_study ()
{
    delete this->d_ptr;
}

const char*
Dcmtk_rt_study::get_ct_series_uid () const
{
    return d_ptr->ct_series_uid;
}

const char*
Dcmtk_rt_study::get_dose_instance_uid () const
{
    return d_ptr->dose_instance_uid;
}

const char*
Dcmtk_rt_study::get_dose_series_uid () const
{
    return d_ptr->dose_series_uid;
}

const char*
Dcmtk_rt_study::get_frame_of_reference_uid () const
{
    return d_ptr->for_uid;
}

const char*
Dcmtk_rt_study::get_rtss_instance_uid () const
{
    return d_ptr->rtss_instance_uid;
}

const char*
Dcmtk_rt_study::get_rtss_series_uid () const
{
    return d_ptr->rtss_series_uid;
}

const char*
Dcmtk_rt_study::get_study_date () const
{
    return d_ptr->date_string.c_str();
}

const char*
Dcmtk_rt_study::get_study_time () const
{
    return d_ptr->time_string.c_str();
}

const char*
Dcmtk_rt_study::get_study_uid () const
{
    return d_ptr->study_uid;
}

std::vector<Dcmtk_slice_data>* 
Dcmtk_rt_study::get_slice_data ()
{
    return d_ptr->slice_data;
}
