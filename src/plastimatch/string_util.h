/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _string_util_h_
#define _string_util_h_

#include "plm_config.h"
#include <string>

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT
int
plm_strcmp (const char* s1, const char* s2);

gpuit_EXPORT
void
string_util_rtrim_whitespace (char *s);

gpuit_EXPORT
int
parse_int13 (int *arr, const char *string);

#if defined __cplusplus
}
#endif

gpuit_EXPORT
const std::string
trim (
    const std::string& str,
    const std::string& whitespace = " \t\r\n"
);

#endif
