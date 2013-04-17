/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rt_study_h_
#define _dcmtk_rt_study_h_

#include "plmbase_config.h"
#include <list>

#include "plm_int.h"
#include "pstring.h"

class Dcmtk_rt_study_private;
class Dcmtk_slice_data;
class Plm_image;
class Volume;

class PLMBASE_API Dcmtk_rt_study {
public:
    Dcmtk_rt_study_private *d_ptr;
public:
    Dcmtk_rt_study ();
    ~Dcmtk_rt_study ();
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
    std::vector<Dcmtk_slice_data>* get_slice_data();
public:
    void set_image (FloatImageType::Pointer image);
};

#endif
