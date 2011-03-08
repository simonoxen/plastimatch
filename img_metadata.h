/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _img_metadata_h_
#define _img_metadata_h_

#include "plm_config.h"
#include <map>
#include "bstrwrap.h"

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
    /* GCS: This is idiotic, but I guess it is what it is.  To avoid string 
       copying, and simultaneously to return the empty string in the 
       event that a key does not exist in the map, I need the equivalent 
       of the string literal "" for std::string.  So here is where it is 
       defined.  N.b. I looked into the possibility of returning 
       std::string and using return value optimization, but that is 
       apparently not possible due to the fact that different strings 
       are returned depending whether the key is found or not. */
    static std::string KEY_NOT_FOUND;

public:
    /* GCS: Note use of unsigned short instead of uint16_t, because of 
       broken stdint implementation in gdcm. */
    std::string
    make_key (unsigned short key1, unsigned short key2) const;
    const std::string&
    get_metadata (const std::string& key) const;
    const std::string&
    get_metadata (unsigned short key1, unsigned short key2) const;
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
	unsigned short elem) const;

public:
    std::map<std::string, std::string> m_data;

public:
    
};

#endif
