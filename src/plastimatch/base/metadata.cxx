/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gdcmFile.h>

#include "plmbase.h"

#include "bstrlib.h"
#include "metadata.h"
#include "make_string.h"

std::string Metadata::KEY_NOT_FOUND = "";

Metadata::Metadata ()
{
    m_parent = 0;
}

Metadata::~Metadata ()
{
}

void
Metadata::create_anonymous ()
{
    /* PatientsName */
    this->set_metadata (0x0010, 0x0010, "ANONYMOUS");
    /* PatientID */
    this->set_metadata (0x0010, 0x0020, dcm_anon_patient_id());
    /* PatientSex */
    this->set_metadata (0x0010, 0x0040, "O");
    /* PatientPosition */
    this->set_metadata (0x0018, 0x5100, "HFS");
}

std::string
Metadata::make_key (unsigned short key1, unsigned short key2) const
{
    return make_string (key1, 4, '0', std::hex) 
	+ ',' + make_string (key2, 4, '0', std::hex);
}

const char*
Metadata::get_metadata_ (const std::string& key) const
{
    std::map<std::string, std::string>::const_iterator it;
    it = m_data.find (key);
    if (it == m_data.end()) {
	/* key not found in map -- check parent */
	if (m_parent) {
	    return m_parent->get_metadata_ (key);
	}
	/* key not found */
	return 0;
    } else {
	/* key found in map */
	return (it->second).c_str();
    }
}

const char*
Metadata::get_metadata_ (unsigned short key1, unsigned short key2) const
{
    return get_metadata_ (make_key (key1, key2));
}

const std::string&
Metadata::get_metadata (const std::string& key) const
{
    std::map<std::string, std::string>::const_iterator it;
    it = m_data.find (key);
    if (it == m_data.end()) {
	/* key not found in map -- check parent */
	if (m_parent) {
	    return m_parent->get_metadata (key);
	}
	/* key not found */
	return KEY_NOT_FOUND;
    } else {
	/* key found in map */
	return it->second;
    }
}

const std::string&
Metadata::get_metadata (unsigned short key1, unsigned short key2) const
{
    return get_metadata (make_key (key1, key2));
}

void
Metadata::set_metadata (const std::string& key, const std::string& val)
{
    //std::cout << "Setting metadata: " << key << " to " << val << std::endl;
    m_data[key] = val;
}

void
Metadata::set_metadata (unsigned short key1, unsigned short key2, 
    const std::string& val)
{
    set_metadata (make_key (key1, key2), val);
}

#if defined (commentout)
void
Metadata::set_from_gdcm_file (
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
Metadata::copy_to_gdcm_file (
    gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
) const
{
    gdcm_file->InsertValEntry (this->get_metadata (group, elem), group, elem);
}
#endif
