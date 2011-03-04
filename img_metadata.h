/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _img_metadata_h_
#define _img_metadata_h_

#include "plm_config.h"
#include <map>
#include "bstrwrap.h"
#include "plm_int.h"

#define MAKE_KEY(a,b) ((uint32_t) (((uint16_t)a) << 16 | ((uint16_t)b)))

class Img_metadata {
public:
    plastimatch1_EXPORT
    Img_metadata ();
    plastimatch1_EXPORT
    ~Img_metadata ();

public:
    enum Keys {
	TEST_KEY = MAKE_KEY(0x0010, 0x0040)
    };

public:
    uint32_t
    make_key (uint16_t key1, uint16_t key2);
    const char* 
    get_metadata (uint32_t key);
    const char* 
    get_metadata (uint16_t key1, uint16_t key2);

    const char* 
    get_patient_name ();

public:
    std::map<uint32_t, const char*> m_data;

    CBString m_patient_name;
    CBString m_patient_id;
    CBString m_patient_sex;

public:
    
};

#endif
