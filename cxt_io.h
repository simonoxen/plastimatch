/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_io_h_
#define _cxt_io_h_

#include "plm_config.h"

class Referenced_dicom_dir;
class Rtss_polyline_set;

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Rtss_polyline_set*
cxt_load_ss_list (Rtss_polyline_set* cxt, const char* xorlist_fn);
plastimatch1_EXPORT
void
cxt_load (
    Rtss_polyline_set *cxt,        /* Output: Structure to load into */
    Referenced_dicom_dir *rdd,     /* Output: Also set some values here */
    const char *cxt_fn             /* Input: file to load from */
);
plastimatch1_EXPORT
void
cxt_save (
    Rtss_polyline_set* cxt,      /* Input: Structure set to save from */
    Referenced_dicom_dir *rdd,   /* Input: Also save some values from here */
    const char* cxt_fn,          /* Input: File to save to */
    bool prune_empty             /* Input: Should we prune empty structures? */
);

#if defined __cplusplus
}
#endif

#endif
