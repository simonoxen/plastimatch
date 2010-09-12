/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_list_io_h_
#define _ss_list_io_h_

#include "plm_config.h"
#include "rtss.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Rtss*
ss_list_load (Rtss* cxt, const char* ss_list_fn);
plastimatch1_EXPORT
void
ss_list_save (Rtss* cxt, const char* cxt_fn);
plastimatch1_EXPORT
void
ss_list_save_colormap (Rtss* cxt, const char* colormap_fn);

#if defined __cplusplus
}
#endif

#endif
