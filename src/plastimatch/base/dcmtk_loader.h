/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_series_set_h_
#define _dcmtk_series_set_h_

#include "plmbase_config.h"
#include <map>
#include <string>

#include "dcmtk_series.h"
#include "itk_image_type.h"

class Rtss_polyline_set;
class Metadata;

class Dcmtk_loader
{
public:
    Dcmtk_loader ();
    Dcmtk_loader (const char* dicom_dir);
    ~Dcmtk_loader ();

public:
    typedef std::map<std::string, Dcmtk_series*> Dcmtk_series_map;
    typedef std::pair<std::string, Dcmtk_series*> Dcmtk_series_map_pair;
    Dcmtk_series_map m_smap;

    Dcmtk_series *ds_rtdose;
    Dcmtk_series *ds_rtss;

    Rtss_polyline_set *cxt;
    Metadata *cxt_metadata;

    Plm_image *img;
    Plm_image *dose;

public:
    void init ();
    void debug (void) const;
    void load_rtss (void);
    void insert_file (const char* fn);
    void insert_directory (const char* fn);
    void parse_directory (void);
    void rtss_load (void);
    void rtdose_load (void);
    void sort_all (void);
};

#endif
