/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_h_
#define _dcmtk_save_h_

#include "plmbase_config.h"

class Dcmtk_save_private;
class Dicom_rt_study;
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
//    void set_cxt (Rtss_structure_set *cxt, Metadata *meta = 0);
    void set_rt_study (Dicom_rt_study *drs);
    void set_cxt (Rtss_structure_set *cxt);
    void set_dose (Volume *vol);
    void set_dose (Volume *vol, Metadata *meta);
    void set_image (Plm_image* img);
public:
    Dcmtk_save_private *d_ptr;
public:
    Rtss_structure_set *cxt;
    Volume *dose;
    Plm_image* img;
public:
    void save (const char *dicom_dir);
    void save_image (const char *dicom_dir);
    void save_rtss (const char *dicom_dir);
    void save_dose (const char *dicom_dir);
    void generate_new_uids ();
};

#endif
