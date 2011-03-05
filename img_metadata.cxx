/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdcmFile.h"
#include "bstrlib.h"
#include "img_metadata.h"

Img_metadata::Img_metadata ()
{
}

Img_metadata::~Img_metadata ()
{
}

std::string
Img_metadata::make_key (unsigned short key1, unsigned short key2)
{
    return std::string (MAKE_KEY (key1,key2));
}

const std::string&
Img_metadata::get_metadata (std::string& key)
{
    return m_data[key];
}

const std::string&
Img_metadata::get_metadata (unsigned short key1, unsigned short key2)
{
    std::string key (MAKE_KEY (key1,key2));
    return get_metadata (key);
}

void
Img_metadata::set_metadata (const std::string& key, const std::string& val)
{
    m_data[key] = val;
}

void
Img_metadata::set_metadata (unsigned short key1, unsigned short key2, 
    const std::string& val)
{
    std::string key (MAKE_KEY (key1,key2));
    set_metadata (key, val);
}

void
Img_metadata::set_from_gdcm_file (
    gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
)
{
    std::string tmp = gdcm_file->GetEntryValue (group, elem);
    if (tmp != gdcm::GDCM_UNFOUND) {
	this->set_metadata (make_key (group, elem), tmp);
    }
}

void
Img_metadata::copy_to_gdcm_file (
    gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
)
{
    gdcm_file->InsertValEntry (this->get_metadata (group, elem), group, elem);
}
