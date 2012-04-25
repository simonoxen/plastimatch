/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include "gdcmBinEntry.h"
#include "gdcmFile.h"
#include "gdcmUtil.h"

#include "metadata.h"

void
gdcm1_get_date_time (
    std::string *date,
    std::string *time
)
{
    *date = gdcm::Util::GetCurrentDate();
    *time = gdcm::Util::GetCurrentTime();
}

void
set_metadata_from_gdcm_file (
    Metadata *meta, 
    /* const */ gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
)
{
    std::string tmp = gdcm_file->GetEntryValue (group, elem);
    if (tmp != gdcm::GDCM_UNFOUND) {
	meta->set_metadata (group, elem, tmp);
    }
}

void
set_gdcm_file_from_metadata (
    gdcm::File *gdcm_file, 
    const Metadata *meta, 
    unsigned short group, 
    unsigned short elem
)
{
    gdcm_file->InsertValEntry (
	meta->get_metadata (group, elem), group, elem);
}
