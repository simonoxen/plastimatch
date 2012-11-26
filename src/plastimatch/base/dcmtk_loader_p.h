/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_loader_p_h_
#define _dcmtk_loader_p_h_

#include "plmbase_config.h"
#include <map>
#include <string>
#include "dcmtk_series.h"

class Dcmtk_rt_study;

/* Map from SeriesInstanceUID to Dcmtk_series */
typedef std::map<std::string, Dcmtk_series*> Dcmtk_series_map;
typedef std::pair<std::string, Dcmtk_series*> Dcmtk_series_map_pair;

class Dcmtk_loader_private {
public:
    Dcmtk_series_map m_smap;
    Dicom_rt_study *m_drs;

public:
    Dcmtk_loader_private () {
        /* Don't create m_drs.  It is set by caller. */
        m_drs = 0;
    }
    ~Dcmtk_loader_private () {
        /* Delete Dicom_series objects in map */
        Dcmtk_series_map::iterator it;
        for (it = m_smap.begin(); it != m_smap.end(); ++it) {
            delete (*it).second;
        }

        /* Don't delete m_drs.  It belongs to caller. */
    }
};

#endif
