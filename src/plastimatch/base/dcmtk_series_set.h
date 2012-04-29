/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_series_set_h_
#define _dcmtk_series_set_h_

#include "plm_config.h"
#include <map>
#include <string>

#include "dcmtk_series.h"

class Rtds;

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
    void load_rtds (Rtds *rtds);
    void insert_file (const char* fn);
    void insert_directory (const char* fn);
    void sort_all (void);
};

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
void
dcmtk_series_set_test (char *dicom_dir);

#if defined __cplusplus
}
#endif

#endif
