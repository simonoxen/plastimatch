/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rt_study_p_h_
#define _dcmtk_rt_study_p_h_

#include "plmbase_config.h"
#include "dicom_rt_study.h"
#include "plm_image.h"

class Dcmtk_series;
class Dcmtk_slice_data;
class Rtss_structure_set;

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

    Dicom_rt_study::Pointer dicom_metadata;

public:
    Dcmtk_rt_study_private ();
    ~Dcmtk_rt_study_private ();
};

#endif
