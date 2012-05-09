/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _simplify_points_h_
#define _simplify_points_h_

#include "plmutil_config.h"

class Rtds;

PLMUTIL_C_API void do_simplify (Rtds *rtds, float percentage);
//void do_simplify (Rtds *rtds, Plm_file_format file_type,int percentage);

#endif
