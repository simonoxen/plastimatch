/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_io_h_
#define _cxt_io_h_

#include "plm_config.h"
#include "rtss.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Rtss*
cxt_load_ss_list (Rtss* cxt, const char* xorlist_fn);
plastimatch1_EXPORT
void
cxt_load (Rtss* cxt, const char* cxt_fn);
plastimatch1_EXPORT
void
cxt_save (Rtss* cxt, const char* cxt_fn, bool prune_empty);

#if defined __cplusplus
}
#endif

#endif
