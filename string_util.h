/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _string_util_h_
#define _string_util_h_

#include "plm_config.h"
#include "plm_int.h"

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
void
string_util_rtrim_whitespace (char *s);

#if defined __cplusplus
}
#endif

#endif
