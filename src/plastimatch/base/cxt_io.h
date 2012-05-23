/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_io_h_
#define _cxt_io_h_

#include "plmbase_config.h"

class Slice_index;
class Rtss;
class Rtss_structure_set;

PLMBASE_C_API Rtss_structure_set* cxt_load_ss_list (
        Rtss_structure_set* cxt,
        const char* xorlist_fn
);
PLMBASE_C_API void cxt_load (
        Rtss *rtss,             /* Output: load into this object */
        Slice_index *rdd,       /* Output: Also set some values here */
        const char *cxt_fn      /* Input: file to load from */
);
PLMBASE_C_API void cxt_save (
        Rtss *rtss,             /* Input: Structure set to save from */
        Slice_index *rdd,       /* Input: Also save some values from here */
        const char* cxt_fn,     /* Input: File to save to */
        bool prune_empty        /* Input: Should we prune empty structures? */
);

#endif
