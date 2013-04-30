/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_io_h_
#define _cxt_io_h_

#include "plmbase_config.h"

class Metadata;
class Rt_study_metadata;
class Rtss;

PLMBASE_C_API Rtss* cxt_load_ss_list (
    Rtss* cxt,
    const char* xorlist_fn
);
PLMBASE_C_API void cxt_load (
    Rtss *cxt,                  /* Output: load into this object */
    Rt_study_metadata *rsm,     /* Output: load into this object */
    const char *cxt_fn          /* Input: file to load from */
);
PLMBASE_C_API void cxt_save (
    Rtss *cxt,                  /* Input: save this object */
    Rt_study_metadata *rsm,     /* Input: save this object */
    const char* cxt_fn,         /* Input: File to save to */
    bool prune_empty            /* Input: Should we prune empty structures? */
);

#endif
