/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_list_io_h_
#define _ss_list_io_h_

#include "plm_config.h"
#include "cxt.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Cxt_structure_list*
ss_list_load (Cxt_structure_list* cxt, const char* ss_list_fn);
plastimatch1_EXPORT
void
ss_list_save (Cxt_structure_list* cxt, const char* cxt_fn);

#if defined __cplusplus
}
#endif

#endif
