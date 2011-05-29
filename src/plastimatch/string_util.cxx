/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "string_util.h"

int
plm_strcmp (const char* s1, const char* s2)
{
    return strncmp (s1, s2, strlen(s2));
}

void
string_util_rtrim_whitespace (char *s)
{
    int len = strlen (s);
    while (len > 0 && isspace(s[len-1])) {
	s[len-1] = 0;
	len--;
    }
}

int
parse_int13 (int *arr, const char *string)
{
    int rc;
    rc = sscanf (string, "%d %d %d", &arr[0], &arr[1], &arr[2]);
    if (rc == 3) {
	return 0;
    } else if (rc == 1) {
	arr[1] = arr[2] = arr[0];
	return 0;
    } else {
	/* Failure */
	return 1;
    }
}

/* String trimming by GMan.
   http://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string/1798170#1798170
   Distributed under Attribution-ShareAlike 3.0 Unported license (CC BY-SA 3.0) 
   http://creativecommons.org/licenses/by-sa/3.0/
*/
const std::string
trim (
    const std::string& str,
    const std::string& whitespace
)
{
    const size_t begin_str = str.find_first_not_of (whitespace);
    if (begin_str == std::string::npos)
    {
        // no content
        return "";
    }

    const size_t end_str = str.find_last_not_of(whitespace);
    const size_t range = end_str - begin_str + 1;

    return str.substr (begin_str, range);
}
