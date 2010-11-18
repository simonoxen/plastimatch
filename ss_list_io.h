/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _ss_list_io_h_
#define _ss_list_io_h_

#include "plm_config.h"
#include "rtss_polyline_set.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT
Rtss_polyline_set*
ss_list_load (Rtss_polyline_set* cxt, const char* ss_list_fn);
plastimatch1_EXPORT
void
ss_list_save (Rtss_polyline_set* cxt, const char* cxt_fn);
plastimatch1_EXPORT
void
ss_list_save_colormap (Rtss_polyline_set* cxt, const char* colormap_fn);

#if defined __cplusplus
}
#endif

#endif
