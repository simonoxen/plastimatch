/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_demographic_h_
#define _xio_demographic_h_

#include "plm_config.h"

class plastimatch1_EXPORT Xio_demographic
{
public:
    CBString m_patient_name;
    CBString m_patient_id;
public:
    Xio_demographic (const char *filename);
    ~Xio_demographic ();
};

#endif
