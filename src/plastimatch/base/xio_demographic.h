/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _xio_demographic_h_
#define _xio_demographic_h_

#include "plmbase_config.h"
#include <string>

class PLMBASE_API Xio_demographic
{
public:
    std::string m_patient_name;
    std::string m_patient_id;
public:
    Xio_demographic (const char *filename);
    ~Xio_demographic ();
};

#endif
