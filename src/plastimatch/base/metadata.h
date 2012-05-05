/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _metadata_h_
#define _metadata_h_

#include "plmbase_config.h"
#include <map>
#include <string>

#if defined (commentout)
namespace gdcm {
    class File;
};
#endif

class API Metadata
{
public:
    Metadata ();
    ~Metadata ();

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
    const char*
    get_metadata_ (const std::string& key) const;
    const char*
    get_metadata_ (unsigned short key1, unsigned short key2) const;
    const std::string&
    get_metadata (const std::string& key) const;
    const std::string&
    get_metadata (unsigned short key1, unsigned short key2) const;
    void
    set_metadata (const std::string& key, const std::string& val);
    void
    set_metadata (unsigned short key1, unsigned short key2,
        const std::string& val);

    void set_parent (Metadata *parent) {
        m_parent = parent;
    }
    void create_anonymous ();

public:
    Metadata *m_parent;
    std::map<std::string, std::string> m_data;

public:
    
};

#endif
