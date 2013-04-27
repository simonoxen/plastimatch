/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_loader_h_
#define _dcmtk_loader_h_

#include "plmbase_config.h"
#include <map>
#include <string>
#include "dicom_rt_study.h"
#include "plm_image.h"
#include "rtss.h"

class Dcmtk_loader_private;
class Dcmtk_series;
class Metadata;
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
    void init ();
    void debug (void) const;
    void set_dicom_metadata (Dicom_rt_study::Pointer drs);
    Metadata *get_metadata ();
    Volume *get_volume ();

    Plm_image::Pointer get_image ();
    Rtss::Pointer get_rtss ();
    Plm_image::Pointer get_dose ();

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
