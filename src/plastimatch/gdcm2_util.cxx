/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmSystem.h"

#include "metadata.h"

void
gdcm2_get_date_time (
    std::string *date,
    std::string *time
)
{
    char datetime[22];
    gdcm::System::GetCurrentDateTime (datetime);
    char c_date[9];
    char c_time[14];
    memcpy (c_date, datetime, 8); c_date[8] = 0;
    memcpy (c_time, datetime+8, 13); c_time[13] = 0;

    *date = c_date;
    *time = c_time;
}

void
set_metadata_from_gdcm_file (
    Metadata *meta, 
    /* const */ gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
)
{
}

void
set_gdcm_file_from_metadata (
    gdcm::File *gdcm_file, 
    const Metadata *meta, 
    unsigned short group, 
    unsigned short elem
)
{
}
