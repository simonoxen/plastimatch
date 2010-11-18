/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _gdcm_rtss_h_
#define _gdcm_rtss_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "rtss_polyline_set.h"

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
    const char *rtss_fn, 
    const char *dicom_dir
);
plastimatch1_EXPORT
void
gdcm_rtss_save (
    Rtss_polyline_set *cxt, 
    char *rtss_fn);

#if defined __cplusplus
}
#endif

#endif
