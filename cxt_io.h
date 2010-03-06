/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_io_h_
#define _cxt_io_h_

#include "plm_config.h"
#include "cxt.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Cxt_structure_list*
cxt_load_ss_list (Cxt_structure_list* structures, const char* xorlist_fn);
plastimatch1_EXPORT
void
cxt_load (Cxt_structure_list* structures, const char* cxt_fn);
plastimatch1_EXPORT
void
cxt_save (Cxt_structure_list* structures, const char* cxt_fn, 
	   bool prune_empty);

#if defined __cplusplus
}
#endif

#endif
