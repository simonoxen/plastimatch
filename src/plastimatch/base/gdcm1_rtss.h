/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_rtss_h_
#define _gdcm1_rtss_h_

#include "plmbase_config.h"

#if GDCM_VERSION_1

class Metadata;
class Rtss_structure_set;
class Slice_index;

PLMBASE_C_API bool gdcm_rtss_probe (const char *rtss_fn);
PLMBASE_C_API void gdcm_rtss_load (
    Rtss_structure_set *cxt,   /* Output: this gets loaded into */
    Metadata *meta,            /* Output: this gets updated too */
    Slice_index *rdd,          /* Output: this gets updated too */
    const char *rtss_fn        /* Input: the file that gets read */
);
PLMBASE_C_API void gdcm_rtss_save (
    Rtss_structure_set *cxt,   /* Input: this is what gets saved */
    Metadata *meta,            /* Input: need to look at this too */
    Slice_index *rdd,          /* Input: need to look at this too */
    char *rtss_fn              /* Input: name of file to write to */
);
#endif /* GDCM_VERSION_1 */

#endif
