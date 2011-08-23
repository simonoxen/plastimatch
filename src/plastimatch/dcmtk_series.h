/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_series_h_
#define _dcmtk_series_h_

#include "plm_config.h"
#include <list>

class Dcmtk_file;

class Dcmtk_series 
{
public:
    Dcmtk_series ();
    ~Dcmtk_series ();

public:
    
    std::list<Dcmtk_file*> m_flist;

public:
    void insert (Dcmtk_file* df);
    void sort (void);
    void debug (void) const;
};

#endif
