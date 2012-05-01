/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_warp_h_
#define _rtds_warp_h_

#include "plm_config.h"
#include "plm_file_format.h"
#include "rtds.h"
#include "warp_parms.h"

plastimatch1_EXPORT
void
rtds_warp (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms);

#endif
