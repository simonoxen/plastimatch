/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_warp_h_
#define _rtds_warp_h_

#include "plmutil_config.h"
#include "base/plm_file_format.h"

class Rtds;
class Warp_parms;

API void rtds_warp (Rtds *rtds, Plm_file_format file_type, Warp_parms *parms);

#endif
