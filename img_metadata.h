/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _img_metadata_h_
#define _img_metadata_h_

#include "plm_config.h"
#include <map>
#include "bstrwrap.h"

#define MAKE_KEY(a,b) (#a "," #b)

namespace gdcm {
    class File;
};

class Img_metadata {
public:
    plastimatch1_EXPORT
    Img_metadata ();
    plastimatch1_EXPORT
    ~Img_metadata ();

public:
    /* GCS: Note use of unsigned short instead of uint16_t, because of 
       broken stdint implementation in gdcm. */
    std::string
    make_key (unsigned short key1, unsigned short key2);
    const std::string&
    get_metadata (std::string& key);
    const std::string&
    get_metadata (unsigned short key1, unsigned short key2);
    void
    set_metadata (const std::string& key, const std::string& val);
    void
    set_metadata (unsigned short key1, unsigned short key2,
	const std::string& val);
    void
    set_from_gdcm_file (gdcm::File *gdcm_file, unsigned short key1, 
	unsigned short key2);
    void
    copy_to_gdcm_file (gdcm::File *gdcm_file, unsigned short group,
	unsigned short elem);

public:
    std::map<std::string, std::string> m_data;

public:
    
};

#endif
