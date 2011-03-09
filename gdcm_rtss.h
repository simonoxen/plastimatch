/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm_rtss_h_
#define _gdcm_rtss_h_

#include "plm_config.h"
#include "cxt_io.h"

class Img_metadata;
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
    Rtss_polyline_set *cxt, 
    Referenced_dicom_dir *rdd, 
    Img_metadata *meta, 
    const char *rtss_fn
);
plastimatch1_EXPORT
void
gdcm_rtss_save (
    Rtss_polyline_set *cxt, 
    Referenced_dicom_dir *rdd, 
    char *rtss_fn
);

#if defined __cplusplus
}
#endif

#endif
