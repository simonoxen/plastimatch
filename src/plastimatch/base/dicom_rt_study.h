/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dicom_rt_study_h_
#define _dicom_rt_study_h_

#include "plmbase_config.h"
#include <list>

#include "plm_int.h"
#include "pstring.h"

class Dicom_rt_study_private;
class Dicom_slice_data;
class Plm_image;
class Volume;

class PLMBASE_API Dicom_rt_study {
public:
    Dicom_rt_study_private *d_ptr;
public:
    Dicom_rt_study ();
    ~Dicom_rt_study ();
public:
    const char* get_ct_series_uid () const;
    const char* get_dose_instance_uid () const;
    const char* get_dose_series_uid () const;
    const char* get_frame_of_reference_uid () const;
    const char* get_rtss_instance_uid () const;
    const char* get_rtss_series_uid () const;
    const char* get_study_date () const;
    const char* get_study_time () const;
    const char* get_study_uid () const;
    std::vector<Dicom_slice_data>* get_slice_data();
};

#endif
