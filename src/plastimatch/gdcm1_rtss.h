/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm1_rtss_h_
#define _gdcm1_rtss_h_

#include "plm_config.h"
#include "cxt_io.h"

class Metadata;
class Rtss_polyline_set;

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
bool
gdcm_rtss_probe (const char *rtss_fn);
plastimatch1_EXPORT
void
gdcm_rtss_load (
    Rtss *rtss,                      /* Output: this gets loaded into */
    Referenced_dicom_dir *rdd,       /* Output: this gets updated too */
    Metadata *meta,              /* Output: this gets updated too */
    const char *rtss_fn              /* Input: the file that gets read */
);
plastimatch1_EXPORT
void
gdcm_rtss_save (
    Rtss *rtss,                    /* Input: this is what gets saved */
    Referenced_dicom_dir *rdd,     /* Input: need to look at this too */
    char *rtss_fn                  /* Input: name of file to write to */
);

#if defined __cplusplus
}
#endif

#endif
