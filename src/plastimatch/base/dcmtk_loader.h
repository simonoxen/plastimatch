/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_loader_h_
#define _dcmtk_loader_h_

#include "plmbase_config.h"
#include <map>
#include <string>

class Dcmtk_loader_private;
class Dcmtk_series;
class Dicom_rt_study;
class Metadata;
class Plm_image;
class Rtss_structure_set;
class Volume;

class Dcmtk_loader
{
public:
    Dcmtk_loader ();
    Dcmtk_loader (const char* dicom_dir);
    ~Dcmtk_loader ();

public:
    Dcmtk_loader_private *d_ptr;

public:
    Dcmtk_series *ds_rtdose;
    Dcmtk_series *ds_rtss;

    Rtss_structure_set *cxt;
    Metadata *cxt_metadata;

    Plm_image *img;
    Plm_image *dose;

public:
    void init ();
    void debug (void) const;
    void set_rt_study (Dicom_rt_study *drs);
    Metadata *get_metadata ();
    Volume *get_volume ();
    void load_rtss (void);
    void insert_file (const char* fn);
    void insert_directory (const char* fn);
    void parse_directory (void);
    void rtss_load (void);
    void rtdose_load (void);
    void sort_all (void);
protected:
    void set_image_uids (const Dcmtk_series *ds);
};

#endif
