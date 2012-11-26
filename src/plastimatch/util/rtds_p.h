/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_p_h_
#define _rtds_p_h_

#include "plmutil_config.h"
#include "dicom_rt_study.h"

class PLMUTIL_API Rtds_private {
public:
    Dicom_rt_study *m_drs;             /* UIDs, etc -- used by dcmtk */

public:
    Rtds_private () {
        m_drs = new Dicom_rt_study;
    }
    ~Rtds_private () {
        delete m_drs;
    }
};

#endif
