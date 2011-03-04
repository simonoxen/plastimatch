/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _img_metadata_h_
#define _img_metadata_h_

#include "plm_config.h"
#include "bstrwrap.h"

class Img_metadata {
public:
    plastimatch1_EXPORT
    Img_metadata ();
    plastimatch1_EXPORT
    ~Img_metadata ();

public:
    CBString m_patient_name;
    CBString m_patient_id;
    CBString m_patient_sex;

public:
    
};

#endif
