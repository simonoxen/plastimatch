/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_h_
#define _dcmtk_save_h_

#include "plmbase_config.h"

class Dcmtk_study_writer;
class Metadata;
class Plm_image;
class Rtss_polyline_set;

class
Dcmtk_save {
public:
    Dcmtk_save ();
    ~Dcmtk_save ();
public:
    void set_cxt (Rtss_polyline_set *cxt, Metadata *cxt_meta);
    void set_dose (Plm_image* img);
    void set_image (Plm_image* img);
public:
    Rtss_polyline_set *cxt;
    Metadata *cxt_meta;
    Plm_image* dose;
    Plm_image* img;
public:
    void save (const char *dicom_dir);
    void save_image (Dcmtk_study_writer *dsw, const char *dicom_dir);
    void save_rtss (Dcmtk_study_writer *dsw, const char *dicom_dir);
};

#endif
