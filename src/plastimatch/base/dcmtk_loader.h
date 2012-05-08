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

class Dcmtk_series_set
{
public:
    Dcmtk_series_set ();
    Dcmtk_series_set (const char* dicom_dir);
    ~Dcmtk_series_set ();

public:
    typedef std::map<std::string, Dcmtk_series*> Dcmtk_series_map;
    typedef std::pair<std::string, Dcmtk_series*> Dcmtk_series_map_pair;
    Dcmtk_series_map m_smap;

public:
    void debug (void) const;
#if defined (GCS_FIX)
    void load_rtds (Rtds *rtds);
#endif
    void insert_file (const char* fn);
    void insert_directory (const char* fn);
    void sort_all (void);
};

#if defined (GCS_FIX)
C_API void dcmtk_series_set_test (char *dicom_dir);
#endif

API ShortImageType::Pointer dcmtk_load (const char *dicom_dir);

#endif
