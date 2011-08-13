/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_file_h_
#define _gdcm1_file_h_

#include "plm_config.h"
#include <string>

namespace gdcm
{
    class File;
};

plastimatch1_EXPORT
std::string
gdcm_file_GetEntryValue (gdcm::File *file, unsigned short group, 
    unsigned short elem);

plastimatch1_EXPORT
const std::string&
gdcm_file_GDCM_UNKNOWN ();

plastimatch1_EXPORT
const std::string&
gdcm_file_GDCM_UNFOUND ();

#endif
