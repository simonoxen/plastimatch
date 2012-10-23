/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_h_
#define _dcmtk_save_h_

#include "plmbase_config.h"

class Dcmtk_rt_study;
class Metadata;
class Plm_image;
class Rtss_structure_set;
class Volume;

class
Dcmtk_save {
public:
    Dcmtk_save ();
    ~Dcmtk_save ();
public:
    void set_cxt (Rtss_structure_set *cxt, Metadata *meta = 0);
    void set_dose (Volume *vol, Metadata *meta = 0);
    void set_image (Plm_image* img);
public:
    Rtss_structure_set *cxt;
    Metadata *cxt_meta;
    Volume *dose;
    Metadata *dose_meta;
    Plm_image* img;
public:
    void save (const char *dicom_dir);
    void save_image (Dcmtk_rt_study *dsw, const char *dicom_dir);
    void save_rtss (Dcmtk_rt_study *dsw, const char *dicom_dir);
    void save_dose (const Dcmtk_rt_study *dsw,
        const char *dicom_dir);
};

#endif
