/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_series_h_
#define _dcmtk_series_h_

#include "plm_config.h"
#include <list>

class Dcmtk_file;
class Plm_image;

class Dcmtk_series 
{
public:
    Dcmtk_series ();
    ~Dcmtk_series ();

public:
    
    std::list<Dcmtk_file*> m_flist;

public:
    void debug (void) const;
    std::string get_modality (void) const;
    std::string get_referenced_uid (void) const;
    void insert (Dcmtk_file* df);
    Plm_image* load_plm_image ();
    void sort (void);
};

#endif
