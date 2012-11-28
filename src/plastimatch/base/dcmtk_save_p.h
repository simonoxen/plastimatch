/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_save_p_h_
#define _dcmtk_save_p_h_

#include "plmbase_config.h"

class Dcmtk_rt_study;

class Dcmtk_save_private {
public:
    Dicom_rt_study *m_drs;

public:
    Dcmtk_save_private () {
        /* Don't create m_drs.  It is set by caller. */
        m_drs = 0;
    }
    ~Dcmtk_save_private () {
        /* Don't delete m_drs.  It belongs to caller. */
    }
};

#endif
