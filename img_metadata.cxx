/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gdcmFile.h"
#include "bstrlib.h"
#include "dicom_util.h"
#include "img_metadata.h"
#include "to_string.h"

std::string Img_metadata::KEY_NOT_FOUND = "";

Img_metadata::Img_metadata ()
{
    /* PatientsName */
    this->set_metadata (0x0010, 0x0010, "ANONYMOUS");
    /* PatientID */
    this->set_metadata (0x0010, 0x0020, make_anon_patient_id());
}

Img_metadata::~Img_metadata ()
{
}

std::string
Img_metadata::make_key (unsigned short key1, unsigned short key2) const
{
    return to_string (key1, 4, '0', std::hex) 
	+ ',' + to_string (key2, 4, '0', std::hex);
}

const std::string&
Img_metadata::get_metadata (const std::string& key) const
{
    std::map<std::string, std::string>::const_iterator it;
    it = m_data.find (key);
    if (it == m_data.end()) {
	/* key not found in map */
	return KEY_NOT_FOUND;
    } else {
	/* key found in map */
	return it->second;
    }
}

const std::string&
Img_metadata::get_metadata (unsigned short key1, unsigned short key2) const
{
    return get_metadata (make_key (key1, key2));
}

void
Img_metadata::set_metadata (const std::string& key, const std::string& val)
{
    std::cout << "Setting metadata: " << key << " to " << val << std::endl;
    m_data[key] = val;
}

void
Img_metadata::set_metadata (unsigned short key1, unsigned short key2, 
    const std::string& val)
{
    set_metadata (make_key (key1, key2), val);
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
) const
{
    gdcm_file->InsertValEntry (this->get_metadata (group, elem), group, elem);
}
