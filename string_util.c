/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "string_util.h"

void
string_util_rtrim_whitespace (char *s)
{
    int len = strlen (s);
    while (len > 0 && isspace(s[len-1])) {
	s[len-1] = 0;
	len--;
    }
}
