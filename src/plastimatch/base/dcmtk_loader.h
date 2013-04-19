/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_loader_h_
#define _dcmtk_loader_h_

#include "plmbase_config.h"
#include <map>
#include <string>
#include "plm_image.h"

class Dcmtk_loader_private;
class Dcmtk_series;
class Dicom_rt_study;
class Metadata;
class Rtss_structure_set;
class Volume;

class Dcmtk_loader
{
public:
    Dcmtk_loader ();
    Dcmtk_loader (const char* dicom_path);
    ~Dcmtk_loader ();

public:
    Dcmtk_loader_private *d_ptr;

public:
    Dcmtk_series *ds_rtdose;
    Dcmtk_series *ds_rtss;

    Rtss_structure_set *cxt;
    Metadata *cxt_metadata;

public:
    void init ();
    void debug (void) const;
    void set_rt_study (Dicom_rt_study *drs);
    Metadata *get_metadata ();
    Volume *get_volume ();

    Plm_image::Pointer get_image ();
    Rtss_structure_set *steal_rtss_structure_set ();
    Plm_image::Pointer get_dose_image ();

//    Plm_image *steal_plm_image ();
//    Plm_image *steal_dose_image ();

    void load_rtss (void);
    void insert_file (const char* fn);
    void insert_directory (const char* fn);
    void parse_directory (void);
    void rtss_load (void);
    void rtdose_load (void);
    void sort_all (void);
protected:
    void set_dose (Plm_image::Pointer dose);
};

#endif
