/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _demographics_h_
#define _demographics_h_

#include "plm_config.h"
#include "bstrwrap.h"

class Demographics {
public:
    plastimatch1_EXPORT
    Demographics ();
    plastimatch1_EXPORT
    ~Demographics ();

public:
    CBString m_patient_name;
    CBString m_patient_id;
    CBString m_patient_sex;

public:
    
};

#if defined __cplusplus
extern "C" {
#endif

#if defined __cplusplus
}
#endif

#endif
