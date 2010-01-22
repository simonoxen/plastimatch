/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cxt_warp_h_
#define _cxt_warp_h_

#include "plm_config.h"
#include "warp_parms.h"

#if defined __cplusplus
extern "C" {
#endif

plastimatch1_EXPORT 
void
cxt_to_mha_write (Cxt_structure_list *structures, 
    Warp_parms *parms);

#if defined __cplusplus
}
#endif

#endif
