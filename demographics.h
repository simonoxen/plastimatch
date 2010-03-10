/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demographics_h_
#define _demographics_h_

#include "plm_config.h"
#include "bstrlib.h"

class Demographics {
public:
    Demographics ();
    ~Demographics ();

public:
    bstring m_patient_name;
    bstring m_patient_id;
    bstring m_patient_sex;
};

#if defined __cplusplus
extern "C" {
#endif

#if defined __cplusplus
}
#endif

#endif
