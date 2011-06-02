/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_util_h_
#define _gdcm1_util_h_

#include "plm_config.h"
#if GDCM_VERSION_1

class Img_metadata;
namespace gdcm { class File; };

void
gdcm1_get_date_time (
    std::string *date,
    std::string *time
);
void
set_metadata_from_gdcm_file (
    Img_metadata *img_metadata, 
    /* const */ gdcm::File *gdcm_file, 
    unsigned short group,
    unsigned short elem
);
void
set_gdcm_file_from_metadata (
    gdcm::File *gdcm_file, 
    const Img_metadata *img_metadata, 
    unsigned short group, 
    unsigned short elem
);

#endif /* GDCM_VERSION_1 */
#endif
