/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_file_h_
#define _gdcm1_file_h_

#include "plmbase_config.h"
#include <string>

namespace gdcm
{
    class File;
};

API std::string
gdcm_file_GetEntryValue (
        gdcm::File *file,
        unsigned short group,
        unsigned short elem
);

API const std::string&
gdcm_file_GDCM_UNKNOWN ();

API const std::string&
gdcm_file_GDCM_UNFOUND ();

#endif
